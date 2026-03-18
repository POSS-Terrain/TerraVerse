import os
import argparse
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from collections import Counter
import clip
import torch.nn.functional as F

# ==============================================================================
# 1. 参数解析与 GPU 设置
# ==============================================================================
parser = argparse.ArgumentParser(description="Vanilla CLIP 地形分类基准测试脚本")
parser.add_argument("--benchmark", type=str, required=True, help="Benchmark 文件夹名称 (例如 benchmark1, benchmark2)")
parser.add_argument("--exp_name", type=str, required=True, help="实验名称 (例如 exp1, exp2)")
parser.add_argument("--gpu", type=str, default="0", help="指定要使用的 GPU ID，如 0 或 1")
parser.add_argument("--ckpt_root", type=str, default="./clip_checkpoints", help="输出结果的根目录")
args, _ = parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# ==============================================================================
# 2. 全局配置与标签定义
# ==============================================================================

EXPERIMENTS = {
    "exp1": {
        # Jackal
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/train/local_image_final"
        ],
        # VAST
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/test"
        ]
    },
    "exp2": {
        # VAST
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/train"
        ],
        # Jackal
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/test/local_image_final"
        ]
    },
    "exp3": {
        # Car： GOOSE, RSCD, RUGD
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/GOOSE/goose/processed_data/train/local_image_final", 
            "/8TBHDD3/tht/TerraData/RSCD/processed_data/train/local_image_final", 
            "/8TBHDD3/tht/TerraData/RUGD/processed_data/train/local_image_final"
        ],
        # AGV： RELLIS, TerraPOSS, Jackal
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/RELLIS/processed_data/test/local_image_final","/8TBHDD3/tht/TerraData/RELLIS/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/test/local_image_final"
        ]
    },
    "exp4": {
        # AGV： RELLIS, TerraPOSS, Jackal
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/RELLIS/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/train/local_image_final"
        ],
        # Car： GOOSE, RSCD, RUGD
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/GOOSE/goose/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/RSCD/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/RUGD/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/RUGD/processed_data/valid/local_image_final"
        ]
    }
}



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
# 3. 数据集类 (保持不变)
# ==============================================================================
class TerraHybridCLIPDataset(Dataset):
    def __init__(self, dir_list, transform=None, max_length=77):
        self.transform = transform
        self.max_length = max_length
        self.samples = [] 
        self.class_counts = Counter() 
        self.target_json = "local_label.json"
        
        print(f"\n📂 [测试加载器] 扫描 {len(dir_list)} 个路径, 目标文件: {self.target_json}")
        
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
        print(f"\n   📊 [测试集] 最终统计:")
        print(f"      - 总图片数: {len(self.samples)}")
        print(f"      - 覆盖类别: 全局 {len(ALL_TERRAIN_CLASSES)} 类中的 {covered_classes} 类")
        
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
# 4. 推理核心逻辑 (大幅简化，仅保留 Vanilla Zero-shot 逻辑)
# ==============================================================================
def evaluate(model, loader, device, all_labels):
    model.eval()
    print("\n正在计算 Vanilla CLIP 零样本推理的文本特征...")
    
    # Vanilla CLIP 只使用基础 Prompt 模板进行零样本分类
    coarse_prompts = [f"The terrain material is {label}." for label in all_labels]
    text_tokens = clip.tokenize(coarse_prompts, truncate=True).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)
        
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        logit_scale = model.logit_scale.exp().item()
        
        for images, _, _, labels in tqdm(loader, desc="Validating (Vanilla CLIP)"):
            images = images.to(device)
            
            # 提取图像特征
            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            
            # 直接计算图像与各类别文本的余弦相似度（Logits）
            logits = logit_scale * (image_features @ text_features.t())
            
            # 取最大相似度的索引作为预测结果
            preds = logits.argmax(dim=-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    return acc, all_targets, all_preds

# ==============================================================================
# 5. 主函数执行
# ==============================================================================
def main():
    if args.exp_name not in EXPERIMENTS:
        raise ValueError(f"❌ 实验名称 '{args.exp_name}' 未在 EXPERIMENTS 中定义！")
        
    exp_cfg = EXPERIMENTS[args.exp_name]
    CONFIG["test_dirs"] = exp_cfg["test_dirs"]

    # 结果输出路径 (加上 vanilla_clip 专属文件夹或命名)
    output_dir = os.path.join(args.ckpt_root, args.benchmark, args.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=======================================================")
    print(f"🚀 启动 Vanilla CLIP 零样本评估: {args.benchmark} | {args.exp_name}")
    print(f"=======================================================\n")
    
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)
    test_transform = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(clip_mean, clip_std)
    ])

    test_dataset = TerraHybridCLIPDataset(CONFIG["test_dirs"], transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)

    print("\n加载基础 Vanilla CLIP 模型 (无需 LoRA 权重)...")
    # 直接加载原始模型，不套用 PeftModel
    model, _ = clip.load(CONFIG["model_name"], device=device)
    model = model.float() 
    model.to(device)

    final_acc, targets, preds = evaluate(model, test_loader, device, ALL_TERRAIN_CLASSES)
    print(f"\n最终盲测准确率 (Vanilla CLIP Accuracy): {final_acc:.4f}")
    
    unique_classes = sorted(list(set(targets)))
    target_names = [ALL_TERRAIN_CLASSES[i] for i in unique_classes]
    
    print("\n" + classification_report(targets, preds, labels=unique_classes, target_names=target_names, digits=4, zero_division=0))
    
    report_dict = classification_report(targets, preds, labels=unique_classes, target_names=target_names, digits=4, zero_division=0, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.loc['Final_Test_Accuracy'] = [final_acc, None, None, None]
    
    # 更改了输出文件名，防止覆盖 TerraCLIP 的结果
    csv_filename = f"vanilla_clip_{args.exp_name}_test_results.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    df_report.to_csv(csv_path, index=True)
    
    print(f"\n📊 [保存成功] 测试结果详细指标已导出至: {csv_path}")

    # ==============================================================================
    # 🚩 汇总所有 Vanilla CLIP 实验的 Final_Test_Accuracy 到总表
    # ==============================================================================
    
    summary_dir = os.path.join(args.ckpt_root, args.benchmark)
    # 使用专门的 Vanilla 汇总表
    summary_csv_path = os.path.join(summary_dir, "vanilla_clip_all_experiments_summary.csv")
    os.makedirs(summary_dir, exist_ok=True)
    
    exp_columns = list(EXPERIMENTS.keys())
    
    if os.path.exists(summary_csv_path):
        df_summary = pd.read_csv(summary_csv_path)
        for col in exp_columns:
            if col not in df_summary.columns:
                df_summary[col] = float('nan')
        df_summary = df_summary[exp_columns]
    else:
        df_summary = pd.DataFrame(columns=exp_columns)
        df_summary.loc[0] = [float('nan')] * len(exp_columns)

    df_summary.at[0, args.exp_name] = final_acc
    df_summary.to_csv(summary_csv_path, index=False, float_format='%.6f')
    
    print(f"📈 [汇总更新] 已将 '{args.exp_name}' 结果录入 Vanilla CLIP 总表: {summary_csv_path}")

if __name__ == "__main__":
    main()