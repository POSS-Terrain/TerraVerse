import os
import argparse
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from collections import Counter
import clip
from peft import PeftModel
import torch.nn.functional as F
import pandas as pd

# ==============================================================================
# 🚨 参数解析与环境变量
# ==============================================================================
parser = argparse.ArgumentParser(description="单测/验证 TerraCLIP 模型")
parser.add_argument("--exp_name", type=str, required=True, help="用于指明测试哪个实验的数据集，如 exp2")
parser.add_argument("--gpu", type=str, default="0", help="指定要使用的 GPU ID")
parser.add_argument("--model_dir", type=str, 
                    # default="/8TBHDD3/tht/TerraX/TerraCLIP/clip_checkpoints/benchmark4/exp2/clip_hybrid_best", 
                    default="/8TBHDD3/tht/TerraX/TerraCLIP/clip_checkpoints/benchmark3_augX/exp2/clip_hybrid_best",
                    help="最佳模型权重的保存路径")
args, _ = parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# ==============================================================================
# PART 0: 实验集配置 (仅保留测试目录配置即可)
# ==============================================================================
EXPERIMENTS = {
    "exp1": {
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/test/local_image_final"
        ]
    },
    "exp2": {
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/dark/local_image_final",
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/dark/local_image_final"
        ]
    },
    "exp3": {
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/sim_fog/local_image_final"
        ]
    },
    "exp4": {
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/snow/local_image_final"
        ]
    },
    "exp5": {
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/sim_sun/local_image_final"
        ]
    }
}

# ==============================================================================
# PART 1: 全局配置与标签定义
# ==============================================================================
ALL_TERRAIN_CLASSES = [
    "asphalt", "brick", "cobble", "concrete", "tile", "coated floor", "flagstone", "board",
    "dirt", "gravel", "mud", "mulch", "rock", "sand", "soil",
    "puddle", "snow", "water", "ice",
    "moss", "grass", "leaves",
]

GLOBAL_LABEL_MAP = {label: idx for idx, label in enumerate(ALL_TERRAIN_CLASSES)}

ALIAS_MAP = {
    "grass floor": "grass", 
    "coated_floor": "coated floor",
}

CONFIG = {
    "batch_size": 512,          
    "num_workers": 8,
    "img_size": 224,            
    "model_name": "ViT-B/32",
}

# ==============================================================================
# PART 2: 数据集类 (精简掉没用的 Train Annotation 解析)
# ==============================================================================
class TerraHybridCLIPDataset(Dataset):
    def __init__(self, dir_list, transform=None, max_length=77):
        self.transform = transform
        self.max_length = max_length
        self.samples = [] 
        self.class_counts = Counter() 
        self.target_json = "local_label.json" # 纯测试只读这个
        
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
        print(f"\n   📊 [Test] 最终统计:")
        print(f"      - 总图片数: {len(self.samples)} 张")
        print(f"      - 覆盖类别: {covered_classes} 类。")

    def _load_test_labels(self, folder_path, json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            items = data.get("items", [])
            valid_count = 0
            
            for item in items:
                img_name = item.get("name")
                label_str = item.get("material")
                
                if img_name and label_str:
                    label_str = label_str.lower()
                    if label_str in ALIAS_MAP: 
                        label_str = ALIAS_MAP[label_str]
                    
                    if label_str in GLOBAL_LABEL_MAP:
                        full_img_path = os.path.join(folder_path, img_name)
                        if not os.path.exists(full_img_path): 
                            continue
                            
                        self.samples.append({
                            "image": full_img_path,
                            "label_idx": GLOBAL_LABEL_MAP[label_str],
                            "coarse_text": "",
                            "fine_text": ""
                        })
                        self.class_counts[label_str] += 1
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
            
        return image, item['coarse_text'], item['fine_text'], item['label_idx']

# ==============================================================================
# PART 3: 测试核心逻辑 (直接借用你最完美的 evaluate 函数)
# ==============================================================================
def evaluate(model, loader, device, all_labels):
    model.eval()
    print("\nComputing multi-granularity text prototypes for Inference (Aligned with Paper)...")
    
    coarse_prompts = [f"The terrain material is {label}." for label in all_labels]
    coarse_tokens = clip.tokenize(coarse_prompts, truncate=True).to(device)
    
    fine_attributes = {
        "weather": ["sunny", "cloudy", "rainy", "foggy", "snowy", "unknown"],
        "lighting": ["strong sunlight", "low light", "shadowed", "dark"],
        "moisture": ["dry", "moist", "wet"],
        "smoothness": ["smooth", "slightly uneven", "severely uneven"],
        "friction_hint": ["high", "medium", "low"],
        "traversability_hint": ["traversable", "non-traversable"]
    }
    
    K = len(all_labels)
    fine_prompts_flat = []
    
    for label in all_labels:
        for attr_key, attr_vals in fine_attributes.items():
            for val in attr_vals:
                fine_prompts_flat.append(f"The terrain material is {label}, characterized by {attr_key} being {val}.")
                
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
        
        for images, _, _, labels in tqdm(loader, desc="Validating (Dynamic Weighting)", leave=False):
            images = images.to(device)
            
            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            
            logits_coarse = logit_scale * (image_features @ coarse_features.t()) # [B, K]
            probs_coarse = F.softmax(logits_coarse, dim=-1) # [B, K]
            
            logits_fine = logit_scale * (image_features @ fine_features_flat.t()) # [B, K*M]
            probs_fine_flat = F.softmax(logits_fine, dim=-1) # [B, K*M]
            
            probs_fine = probs_fine_flat.view(-1, K, M)
            
            sum_score = probs_fine.sum(dim=-1) # [B, K]
            max_score = probs_fine.max(dim=-1).values # [B, K]
            fine_score = sum_score + lambda_weight * max_score # [B, K]
            
            import numpy as np
            entropy = -torch.sum(probs_coarse * torch.log(probs_coarse + 1e-9), dim=-1) # [B]
            alpha = entropy / np.log(K) 
            alpha = alpha.unsqueeze(-1) # [B, 1]
            
            final_scores = (1.0 - alpha) * probs_coarse + alpha * fine_score # [B, K]
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
    
    test_dirs = EXPERIMENTS[args.exp_name]["test_dirs"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=======================================================")
    print(f"🚀 启动纯测试 | 评估实验: {args.exp_name} | GPU: {args.gpu}")
    print(f"📂 读取权重从: {args.model_dir}")
    print(f"=======================================================\n")
    
    # 1. 加载模型
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"❌ 找不到模型权重目录: {args.model_dir}")

    print("Loading Base CLIP Model...")
    base_model, _ = clip.load(CONFIG["model_name"], device=device)
    base_model = base_model.float() 
    
    print("Loading LoRA Weights...")
    model = PeftModel.from_pretrained(base_model, args.model_dir)
    model.to(device)
    print("✅ 已成功加载最佳模型组合")
    
    # 2. 图像预处理
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)
    valid_transform_safe = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(clip_mean, clip_std)
    ])

    # 3. 加载测试数据集
    test_dataset = TerraHybridCLIPDataset(test_dirs, transform=valid_transform_safe)
    test_loader  = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)

    # 4. 执行推理评估
    final_acc, targets, preds = evaluate(model, test_loader, device, ALL_TERRAIN_CLASSES)
    print(f"\n🏆 Final Test Accuracy ({args.exp_name}): {final_acc:.4f}\n")
    
    # 5. 打印并保存报告
    unique_classes = sorted(list(set(targets)))
    target_names = [ALL_TERRAIN_CLASSES[i] for i in unique_classes]
    
    print(classification_report(targets, preds, labels=unique_classes, target_names=target_names, digits=4, zero_division=0))
    
    report_dict = classification_report(targets, preds, labels=unique_classes, target_names=target_names, digits=4, zero_division=0, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.loc['Final_Test_Accuracy'] = [final_acc, None, None, None]
    
    csv_filename = f"PURE_TEST_RESULTS_{args.exp_name}.csv"
    df_report.to_csv(csv_filename, index=True)
    
    print(f"\n📊 [保存成功] 测试结果已导出至当前目录: {csv_filename}")

if __name__ == "__main__":
    main(args)