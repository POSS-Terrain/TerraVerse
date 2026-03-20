import os
import argparse

# ==============================================================================
# 🚨 极其重要：必须在导入 torch 前完成参数解析和 GPU 环境变量设置！
# ==============================================================================
parser = argparse.ArgumentParser(description="TerraCLIP (LoRA) 纯测试脚本")
parser.add_argument("--exp_name", type=str, required=True, help="实验名称，如 exp1, exp2")
parser.add_argument("--gpu", type=str, default="0", help="指定要使用的 GPU ID，如 0 或 1")
parser.add_argument("--ckpt_path", type=str, 
                    default="/8TBHDD3/tht/TerraX/TerraCLIP/clip_checkpoints/benchmark6/exp2/clip_traversable_best", 
                    help="训练好的 LoRA 权重目录路径 (包含 adapter_model.bin 等文件)")
args, _ = parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# ==============================================================================
# --- 环境设置完毕后，再导入深度学习包 ---
# ==============================================================================
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from collections import Counter
import clip
import pandas as pd
from peft import PeftModel

# ==============================================================================
# PART 0: 实验集配置 (仅保留测试集)
# ==============================================================================
EXPERIMENTS = {
    "exp1": {
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/ORFD/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/ORFD/processed_data/valid/local_image_final"
        ]
    },
    "exp2": {
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/ORAD-3D/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/ORAD-3D/processed_data/valid/local_image_final"
        ]
    }
}

# ==============================================================================
# PART 1: 全局配置与标签定义
# ==============================================================================
TRAVERSABILITY_CLASSES = ["non-traversable", "traversable"]
GLOBAL_LABEL_MAP = {label: idx for idx, label in enumerate(TRAVERSABILITY_CLASSES)}

CONFIG = {
    "batch_size": 256,          # 测试时可根据显存适当调整
    "num_workers": 8,
    "img_size": 224,            
    "model_name": "ViT-B/32",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# ==============================================================================
# PART 2: 纯测试版数据集类
# ==============================================================================
class TerraInferenceCLIPDataset(Dataset):
    def __init__(self, dir_list, transform=None):
        self.transform = transform
        self.samples = [] 
        self.class_counts = Counter() 
        self.target_json = "local_label.json"
        
        print(f"\n📂 [Test Loader] 扫描 {len(dir_list)} 个路径, 目标文件: {self.target_json}")
        
        for folder_path in dir_list:
            if not os.path.exists(folder_path):
                print(f"   ❌ [跳过] 路径不存在: {folder_path}")
                continue
            
            json_path = os.path.join(folder_path, self.target_json)
            if os.path.exists(json_path):
                self._load_test_labels(folder_path, json_path)
            else:
                print(f"   ❌ [跳过] 未找到 {self.target_json}: {folder_path}")

        covered_classes = len(self.class_counts.keys())
        print(f"\n   📊 [Test Dataset] 最终统计:")
        print(f"      - 总图片数: {len(self.samples)} 张")
        print(f"      - 覆盖类别: 全局 {len(TRAVERSABILITY_CLASSES)} 类中的 {covered_classes} 类。")

    def _load_test_labels(self, folder_path, json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            items = data.get("items", [])
            valid_count = 0
            
            for item in items:
                img_name = item.get("name")
                trav_str = item.get("traversability", "").lower()
                
                if not img_name or not trav_str: continue
                
                if trav_str in GLOBAL_LABEL_MAP:
                    full_img_path = os.path.join(folder_path, img_name)
                    if not os.path.exists(full_img_path): continue
                        
                    self.samples.append({
                        "image": full_img_path,
                        "label_idx": GLOBAL_LABEL_MAP[trav_str]
                    })
                    self.class_counts[trav_str] += 1
                    valid_count += 1
            print(f"      -> {json_path.split('/')[-3]}: 成功加载 {valid_count} 张 (GT Label)")
        except Exception as e:
            print(f"      -> [Error] 解析 {json_path} 失败: {e}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        try:
            image = Image.open(item['image']).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        return image, item['label_idx']

# ==============================================================================
# PART 3: 推理核心逻辑 (动态权重多粒度评估)
# ==============================================================================
def run_inference(model, loader, device, all_labels):
    model.eval()
    print("\n🧠 Computing multi-granularity text prototypes for Inference...")
    
    # 1. 粗粒度文本
    coarse_prompts = [f"The traversability of material is {label}." for label in all_labels]
    coarse_tokens = clip.tokenize(coarse_prompts, truncate=True).to(device)
    
    # 2. 细粒度属性池
    fine_attributes = {
        "material": ["asphalt", "brick", "cobble", "concrete", "tile", "coated floor", "flagstone", "board", "dirt", "gravel", "mud", "mulch", "rock", "sand", "soil", "puddle", "snow", "water", "ice", "moss", "grass", "leaves"],
        "weather": ["sunny", "cloudy", "rainy", "foggy", "snowy", "unknown"],
        "lighting": ["strong sunlight", "low light", "shadowed", "dark"],
        "moisture": ["dry", "moist", "wet"],
        "smoothness": ["smooth", "slightly uneven", "severely uneven"],
        "friction_hint": ["high", "medium", "low"]
    }
    
    K = len(all_labels)
    fine_prompts_flat = []
    
    for label in all_labels:
        for attr_key, attr_vals in fine_attributes.items():
            for val in attr_vals:
                fine_prompts_flat.append(f"The traversability of material is {label}, characterized by {attr_key} being {val}.")
                
    M = len(fine_prompts_flat) // K  
    
    with torch.no_grad():
        coarse_features = model.encode_text(coarse_tokens)
        coarse_features = F.normalize(coarse_features, dim=-1)
        
        fine_tokens = clip.tokenize(fine_prompts_flat, truncate=True).to(device)
        fine_features_flat = model.encode_text(fine_tokens)
        fine_features_flat = F.normalize(fine_features_flat, dim=-1)
        
    all_preds = []
    all_targets = []
    lambda_weight = 0.3 
    
    with torch.no_grad():
        logit_scale = model.logit_scale.exp().item()
        
        for images, labels in tqdm(loader, desc="Validating (Dynamic Weighting)"):
            images = images.to(device)
            
            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            
            logits_coarse = logit_scale * (image_features @ coarse_features.t()) 
            probs_coarse = F.softmax(logits_coarse, dim=-1) 
            
            logits_fine = logit_scale * (image_features @ fine_features_flat.t()) 
            probs_fine_flat = F.softmax(logits_fine, dim=-1) 
            
            probs_fine = probs_fine_flat.view(-1, K, M)
            
            sum_score = probs_fine.sum(dim=-1) 
            max_score = probs_fine.max(dim=-1).values 
            fine_score = sum_score + lambda_weight * max_score 
            
            import numpy as np 
            entropy = -torch.sum(probs_coarse * torch.log(probs_coarse + 1e-9), dim=-1) 
            
            alpha = entropy / np.log(K) 
            alpha = alpha.unsqueeze(-1) 
            
            final_scores = (1.0 - alpha) * probs_coarse + alpha * fine_score 
            preds = final_scores.argmax(dim=-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    return acc, all_targets, all_preds

# ==============================================================================
# PART 4: 主函数
# ==============================================================================
def main(args):
    if args.exp_name not in EXPERIMENTS:
        raise ValueError(f"❌ 实验名称 '{args.exp_name}' 未在 EXPERIMENTS 中定义！")
    
    CONFIG["test_dirs"] = EXPERIMENTS[args.exp_name]["test_dirs"]
    device = CONFIG["device"]
    
    print(f"\n=======================================================")
    print(f"🚀 启动 TerraCLIP 测试: {args.exp_name} | 使用设备: GPU {args.gpu}")
    print(f"📦 目标模型路径: {args.ckpt_path}")
    print(f"=======================================================\n")
    
    # 1. 初始化预处理 Transform
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)
    
    test_transform = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(clip_mean, clip_std)
    ])

    # 2. 加载数据集
    test_dataset = TerraInferenceCLIPDataset(CONFIG["test_dirs"], transform=test_transform)
    test_loader  = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
    
    # 3. 加载基础模型与 LoRA 权重
    print(f"\n🏗️ Loading Base CLIP Model ({CONFIG['model_name']})...")
    base_model, _ = clip.load(CONFIG["model_name"], device=device)
    base_model = base_model.float() 
    
    if os.path.exists(args.ckpt_path):
        print(f"🔗 Injecting LoRA Weights from: {args.ckpt_path} ...")
        model = PeftModel.from_pretrained(base_model, args.ckpt_path)
        model.to(device)
        print("✅ 权重加载成功！")
    else:
        raise FileNotFoundError(f"❌ 未找到 LoRA 权重目录: {args.ckpt_path}")

    # 4. 执行盲测
    print("\n=== 🏆 开始进行盲测 (Testing) ===")
    final_acc, targets, preds = run_inference(model, test_loader, device, TRAVERSABILITY_CLASSES)
    
    print(f"\nFinal Test Accuracy: {final_acc:.4f}")
    
    unique_classes = sorted(list(set(targets)))
    target_names = [TRAVERSABILITY_CLASSES[i] for i in unique_classes]
    
    print("\n" + classification_report(targets, preds, labels=unique_classes, target_names=target_names, digits=4, zero_division=0))
    
    # 5. 导出结果
    report_dict = classification_report(targets, preds, labels=unique_classes, target_names=target_names, digits=4, zero_division=0, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.loc['Final_Test_Accuracy'] = [final_acc, None, None, None]
    
    csv_filename = f"{args.exp_name}_terraclip_inference_results.csv"
    df_report.to_csv(csv_filename, index=True)
    
    print(f"\n📊 [保存成功] 测试结果详细指标已导出至当前目录: ./{csv_filename}")

if __name__ == "__main__":
    main(args)