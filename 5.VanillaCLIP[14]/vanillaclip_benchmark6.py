import os
import argparse

# ==============================================================================
# 1. 🚨 必须在导入 torch 前完成参数解析和 GPU 环境变量设置！
# ==============================================================================
parser = argparse.ArgumentParser(description="Vanilla CLIP 可通行性(二分类)零样本测试")
parser.add_argument("--benchmark", type=str, default="benchmark6", help="Benchmark 文件夹名称")
parser.add_argument("--exp_name", type=str, required=True, help="实验名称，如 exp1, exp2")
parser.add_argument("--gpu", type=str, default="0", help="指定要使用的 GPU ID，如 0 或 1")
parser.add_argument("--ckpt_root", type=str, default="./clip_checkpoints", help="输出结果的根目录")
args, _ = parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# ==============================================================================
# --- 环境设置完毕后，再导入深度学习包 ---
# ==============================================================================
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
# PART 0: 实验集配置 (仅保留测试所需目录)
# ==============================================================================
EXPERIMENTS = {
    "exp1": {
        # ORFD
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/ORFD/processed_data/train/local_image_final"
        ],
        
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/ORFD/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/ORFD/processed_data/valid/local_image_final"
        ]
    },
    "exp2": {
        # ORAD-3D
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/ORAD-3D/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/ORAD-3D/processed_data/train_1/local_image_final",
            "/8TBHDD3/tht/TerraData/ORAD-3D/processed_data/train_2/local_image_final",
            "/8TBHDD3/tht/TerraData/ORAD-3D/processed_data/train_3/local_image_final",
            "/8TBHDD3/tht/TerraData/ORAD-3D/processed_data/train_4/local_image_final",
            "/8TBHDD3/tht/TerraData/ORAD-3D/processed_data/train_5/local_image_final"
        ],
        
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/ORAD-3D/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/ORAD-3D/processed_data/valid/local_image_final"
        ]
    }
}


# ==============================================================================
# PART 1: 全局配置与二分类标签定义
# ==============================================================================
TRAVERSABILITY_CLASSES = ["non-traversable", "traversable"]
GLOBAL_LABEL_MAP = {label: idx for idx, label in enumerate(TRAVERSABILITY_CLASSES)}

CONFIG = {
    "batch_size": 512,          
    "num_workers": 8,
    "img_size": 224,            
    "model_name": "ViT-B/32",
}

# ==============================================================================
# PART 2: 测试数据集类
# ==============================================================================
class TerraVanillaCLIPDataset(Dataset):
    def __init__(self, dir_list, transform=None):
        self.transform = transform
        self.samples = [] 
        self.class_counts = Counter() 
        self.target_json = "local_label.json"
        
        print(f"\n📂 [测试集 Loader] 扫描 {len(dir_list)} 个路径, 目标文件: {self.target_json}")
        
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
        print(f"      - 总图片数: {len(self.samples)} 张")
        print(f"      - 覆盖类别: 全局 {len(TRAVERSABILITY_CLASSES)} 类中的 {covered_classes} 类")
        
    def _load_test_labels(self, folder_path, json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            items = data.get("items", [])
            valid_count = 0
            
            for item in items:
                img_name = item.get("name")
                
                # 💡 核心修改点：直接读取 JSON 中的 traversability 字段
                trav_str = item.get("traversability", "").lower()
                
                if not img_name or not trav_str: 
                    continue
                
                # 只要是 'traversable' 或 'non-traversable' 就记录
                if trav_str in GLOBAL_LABEL_MAP:
                    full_img_path = os.path.join(folder_path, img_name)
                    if not os.path.exists(full_img_path): 
                        continue
                        
                    self.samples.append({
                        "image": full_img_path,
                        "label_idx": GLOBAL_LABEL_MAP[trav_str]
                    })
                    self.class_counts[trav_str] += 1
                    valid_count += 1
                    
            print(f"      -> {json_path.split('/')[-3]}: 成功加载 {valid_count} 张 (Traversability Label)")
        except Exception as e:
            print(f"      -> [Error] 解析 {json_path} 失败: {e}")

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        try:
            image = Image.open(item['image']).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        # Vanilla 只需要返回图片和索引即可
        return image, item['label_idx']

# ==============================================================================
# PART 3: 零样本推理核心逻辑
# ==============================================================================
def evaluate(model, loader, device, all_labels):
    model.eval()
    print("\n正在计算 Vanilla CLIP 零样本推理的文本特征...")
    
    # 构造通顺的英文 Prompt
    prompts = [f"This terrain is {label}." for label in all_labels]
    text_tokens = clip.tokenize(prompts, truncate=True).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)
        
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        logit_scale = model.logit_scale.exp().item()
        
        for images, labels in tqdm(loader, desc="Validating (Vanilla Zero-Shot)"):
            images = images.to(device)
            
            # 提取图像特征并归一化
            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            
            # 计算余弦相似度 (Logits)
            logits = logit_scale * (image_features @ text_features.t())
            
            # 取最大值作为预测类别
            preds = logits.argmax(dim=-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    return acc, all_targets, all_preds

# ==============================================================================
# PART 4: 主函数执行
# ==============================================================================
def main():
    if args.exp_name not in EXPERIMENTS:
        raise ValueError(f"❌ 实验名称 '{args.exp_name}' 未在 EXPERIMENTS 中定义！")
        
    exp_cfg = EXPERIMENTS[args.exp_name]
    CONFIG["test_dirs"] = exp_cfg["test_dirs"]

    output_dir = os.path.join(args.ckpt_root, args.benchmark, args.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=======================================================")
    print(f"🚀 启动 Vanilla CLIP [二分类可通行性] 评估 | 目标: {args.benchmark} -> {args.exp_name}")
    print(f"=======================================================\n")
    
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)
    test_transform = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(clip_mean, clip_std)
    ])

    test_dataset = TerraVanillaCLIPDataset(CONFIG["test_dirs"], transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)

    print("\n加载基础 Vanilla CLIP 模型 (完全未微调)...")
    model, _ = clip.load(CONFIG["model_name"], device=device)
    model = model.float() 
    model.to(device)

    final_acc, targets, preds = evaluate(model, test_loader, device, TRAVERSABILITY_CLASSES)
    print(f"\n最终盲测准确率 (Vanilla CLIP Traversability Acc): {final_acc:.4f}")
    
    unique_classes = sorted(list(set(targets)))
    target_names = [TRAVERSABILITY_CLASSES[i] for i in unique_classes]
    
    print("\n" + classification_report(targets, preds, labels=unique_classes, target_names=target_names, digits=4, zero_division=0))
    
    report_dict = classification_report(targets, preds, labels=unique_classes, target_names=target_names, digits=4, zero_division=0, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.loc['Final_Test_Accuracy'] = [final_acc, None, None, None]
    
    # 输出详细 CSV
    csv_filename = f"vanilla_clip_traversability_{args.exp_name}_test_results.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    df_report.to_csv(csv_path, index=True)
    print(f"\n📊 [保存成功] 测试结果已导出至: {csv_path}")

    # ==============================================================================
    # 汇总所有 Vanilla CLIP 二分类实验的结果
    # ==============================================================================
    summary_dir = os.path.join(args.ckpt_root, args.benchmark)
    summary_csv_path = os.path.join(summary_dir, "vanilla_clip_traversability_summary.csv")
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
    print(f"📈 [汇总更新] 结果已录入总表: {summary_csv_path}")

if __name__ == "__main__":
    main()