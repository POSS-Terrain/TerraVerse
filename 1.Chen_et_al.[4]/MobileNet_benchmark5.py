import os
import argparse

# ==============================================================================
# 🚨 极其重要：必须在导入 torch 前完成参数解析和 GPU 环境变量设置！
# ==============================================================================
parser = argparse.ArgumentParser(description="多卡并发训练 MobileNetV2 - Few-Shot Adaptation")
parser.add_argument("--exp_name", type=str, required=True, help="实验名称，如 exp1, exp2")
parser.add_argument("--gpu", type=str, default="0", help="指定要使用的 GPU ID，如 0 或 1")
args, _ = parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# ==============================================================================
# --- 环境设置完毕后，再导入深度学习包 ---
# ==============================================================================
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from collections import Counter
import pandas as pd
import numpy as np

# ==============================================================================
# PART 0: 实验集配置 (EXPERIMENTS) 
# ==============================================================================
# 为了做 Few-Shot，需要：源域训练集(Base Train)、目标域训练集(抽1% Fine-tune)、目标域测试集(Test)
base_path = "/8TBHDD3/tht/TerraData"

EXPERIMENTS = {
    "exp1": { # PO->JA
        "source_train_dirs": [f"{base_path}/TerraPOSS/processed_data/train/local_image_final"],
        "target_train_dirs": [f"{base_path}/Jackal/processed_data/train/local_image_final"],
        "target_test_dirs": [f"{base_path}/Jackal/processed_data/test/local_image_final"]
    },
    "exp2": { # RS->JA
        "source_train_dirs": [f"{base_path}/RSCD/processed_data/train/local_image_final"],
        "target_train_dirs": [f"{base_path}/Jackal/processed_data/train/local_image_final"],
        "target_test_dirs": [f"{base_path}/Jackal/processed_data/test/local_image_final"]
    },
    "exp3": { # JA->PO
        "source_train_dirs": [f"{base_path}/Jackal/processed_data/train/local_image_final"],
        "target_train_dirs": [f"{base_path}/TerraPOSS/processed_data/train/local_image_final"],
        "target_test_dirs": [f"{base_path}/TerraPOSS/processed_data/test/local_image_final"]
    },
    "exp4": { # RS->PO
        "source_train_dirs": [f"{base_path}/RSCD/processed_data/train/local_image_final"],
        "target_train_dirs": [f"{base_path}/TerraPOSS/processed_data/train/local_image_final"],
        "target_test_dirs": [f"{base_path}/TerraPOSS/processed_data/test/local_image_final"]
    },
    "exp5": { # JA->RS
        "source_train_dirs": [f"{base_path}/Jackal/processed_data/train/local_image_final"],
        "target_train_dirs": [f"{base_path}/RSCD/processed_data/train/local_image_final"],
        "target_test_dirs": [f"{base_path}/RSCD/processed_data/test/local_image_final"]
    },
    "exp6": { # PO->RS
        "source_train_dirs": [f"{base_path}/TerraPOSS/processed_data/train/local_image_final"],
        "target_train_dirs": [f"{base_path}/RSCD/processed_data/train/local_image_final"],
        "target_test_dirs": [f"{base_path}/RSCD/processed_data/test/local_image_final"]
    },
    "exp7": { # JA->VA
        "source_train_dirs": [f"{base_path}/Jackal/processed_data/train/local_image_final"],
        "target_train_dirs": [f"{base_path}/VAST/processed_data/local_image_final_split/train"],
        "target_test_dirs": [f"{base_path}/VAST/processed_data/local_image_final_split/test"]
    },
    "exp8": { # RS->VA
        "source_train_dirs": [f"{base_path}/RSCD/processed_data/train/local_image_final"],
        "target_train_dirs": [f"{base_path}/VAST/processed_data/local_image_final_split/train"],
        "target_test_dirs": [f"{base_path}/VAST/processed_data/local_image_final_split/test"]
    },
    "exp9": { # PO->VA
        "source_train_dirs": [f"{base_path}/TerraPOSS/processed_data/train/local_image_final"],
        "target_train_dirs": [f"{base_path}/VAST/processed_data/local_image_final_split/train"],
        "target_test_dirs": [f"{base_path}/VAST/processed_data/local_image_final_split/test"]
    },
    "exp10": { # VA->JA
        "source_train_dirs": [f"{base_path}/VAST/processed_data/local_image_final_split/train"],
        "target_train_dirs": [f"{base_path}/Jackal/processed_data/train/local_image_final"],
        "target_test_dirs": [f"{base_path}/Jackal/processed_data/test/local_image_final"]
    },
    "exp11": { # VA->RS
        "source_train_dirs": [f"{base_path}/VAST/processed_data/local_image_final_split/train"],
        "target_train_dirs": [f"{base_path}/RSCD/processed_data/train/local_image_final"],
        "target_test_dirs": [f"{base_path}/RSCD/processed_data/test/local_image_final"]
    },
    "exp12": { # VA->PO
        "source_train_dirs": [f"{base_path}/VAST/processed_data/local_image_final_split/train"],
        "target_train_dirs": [f"{base_path}/TerraPOSS/processed_data/train/local_image_final"],
        "target_test_dirs": [f"{base_path}/TerraPOSS/processed_data/test/local_image_final"]
    }
}

# ==========================================
# PART 1: 全局标签定义 & 配置参数 
# ==========================================
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
    "json_name": "local_label.json",
    "batch_size": 256,
    "lr": 1e-3,
    "base_epochs": 80, 
    "fewshot_epochs": 50, # 论文中提到的 fine-tune 50 epochs
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 8,
    "img_size": 224,
    "save_dir": "./mobilenet_checkpoints/benchmark5" 
}

# ==========================================
# PART 2: 数据集类与 Few-Shot 采样器
# ==========================================
class TerraHybridDataset(Dataset):
    def __init__(self, dir_list, json_name, transform=None, is_train=False):
        self.transform = transform
        self.samples = [] 
        self.class_counts = Counter() 
        self.dataset_type = "Train/Valid (Raw)" if is_train else "Test"
        
        for folder_path in dir_list:
            if not os.path.exists(folder_path): continue
            json_path = os.path.join(folder_path, json_name)
            if os.path.exists(json_path):
                self._load_json_mode(folder_path, json_path)
            else:
                self._load_folder_mode(folder_path)

        if len(self.samples) == 0:
            raise ValueError(f"❌ 未加载到 {self.dataset_type} 数据！请检查路径。")

    def _load_json_mode(self, folder_path, json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for item in data.get("items", []):
                img_name = item.get("name")
                label_str = item.get("label", "").lower()
                if img_name and label_str:
                    if label_str in ALIAS_MAP: label_str = ALIAS_MAP[label_str]
                    if label_str in GLOBAL_LABEL_MAP:
                        full_img_path = os.path.join(folder_path, img_name)
                        if os.path.exists(full_img_path):
                            self.samples.append((full_img_path, label_str))
                            self.class_counts[label_str] += 1
        except Exception as e: pass

    def _load_folder_mode(self, root_path):
        for class_name in os.listdir(root_path):
            class_dir = os.path.join(root_path, class_name)
            if not os.path.isdir(class_dir): continue
            label_str = class_name.lower()
            if label_str in ALIAS_MAP: label_str = ALIAS_MAP[label_str]
            if label_str in GLOBAL_LABEL_MAP:
                files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                for img_file in files:
                    full_img_path = os.path.join(class_dir, img_file)
                    self.samples.append((full_img_path, label_str))
                    self.class_counts[label_str] += 1

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_str = self.samples[idx]
        label_idx = GLOBAL_LABEL_MAP[label_str]
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224))
        
        if self.transform:
            image = self.transform(image)
        return image, label_idx

class DatasetWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.subset[index]
        if self.transform: image = self.transform(image)
        return image, label

    def __len__(self): return len(self.subset)

# 🌟 新增：分层抽样出 1% 的数据用于 Few-Shot Fine-Tuning
def get_stratified_few_shot_subset(dataset, fraction=0.01):
    labels = [dataset.samples[i][1] for i in range(len(dataset))]
    unique_labels = list(set(labels))
    selected_indices = []
    
    np.random.seed(42) # 固定随机种子以保证实验可复现
    for lbl in unique_labels:
        lbl_indices = [i for i, x in enumerate(labels) if x == lbl]
        # 确保每个类别至少抽取1个样本
        n_select = max(1, int(len(lbl_indices) * fraction)) 
        selected = np.random.choice(lbl_indices, n_select, replace=False)
        selected_indices.extend(selected)
        
    print(f"   🔍 Few-Shot 1% 采样完成: 共抽取了 {len(selected_indices)} 张图片。")
    return Subset(dataset, selected_indices)

# ==========================================
# PART 3: MobileNetV2 模型
# ==========================================
class VisionNet(nn.Module):
    def __init__(self, num_classes, head_type="wide"):
        super(VisionNet, self).__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        self.backbone = models.mobilenet_v2(weights=weights)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        in_features = self.backbone.classifier[1].in_features 
        
        if head_type == "wide":
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.3), 
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(512, num_classes) 
            )
        else:
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features, num_classes)
            )

    def forward(self, x): return self.backbone(x)

# ==========================================
# PART 4: 主程序 (Base Training + 1% Fine-Tuning)
# ==========================================
def main(args):
    if args.exp_name not in EXPERIMENTS:
        raise ValueError(f"❌ 实验名称 '{args.exp_name}' 未在 EXPERIMENTS 中定义！")
    
    exp_cfg = EXPERIMENTS[args.exp_name]
    
    exp_save_dir = os.path.join(CONFIG["save_dir"], args.exp_name)
    os.makedirs(exp_save_dir, exist_ok=True)
    best_base_model_path = os.path.join(exp_save_dir, "mobilenet_base_best.pth")
    best_fs_model_path = os.path.join(exp_save_dir, "mobilenet_fewshot_best.pth")

    print(f"\n=======================================================")
    print(f"🚀 启动 Few-Shot 实验: {args.exp_name} | GPU: {args.gpu}")
    print(f"=======================================================\n")
    
    train_transforms = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ------------------ 阶段 1：数据准备 ------------------
    # 1. Base 训练集 (Source)
    full_source_dataset = TerraHybridDataset(exp_cfg["source_train_dirs"], CONFIG["json_name"], transform=None, is_train=True)
    total_size = len(full_source_dataset)
    train_size = int(0.8 * total_size)
    valid_size = total_size - train_size
    raw_train_set, raw_valid_set = random_split(full_source_dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(42))
    
    source_train_dataset = DatasetWrapper(raw_train_set, transform=train_transforms)
    source_valid_dataset = DatasetWrapper(raw_valid_set, transform=test_transforms)

    # 2. Few-Shot 目标域训练集 (1% Target)
    full_target_train_dataset = TerraHybridDataset(exp_cfg["target_train_dirs"], CONFIG["json_name"], transform=None, is_train=True)
    few_shot_raw_subset = get_stratified_few_shot_subset(full_target_train_dataset, fraction=0.01)
    target_fewshot_dataset = DatasetWrapper(few_shot_raw_subset, transform=train_transforms)

    # 3. 目标域测试集 (Target Test)
    target_test_dataset = TerraHybridDataset(exp_cfg["target_test_dirs"], CONFIG["json_name"], transform=test_transforms, is_train=False)

    source_train_loader = DataLoader(source_train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)
    source_valid_loader = DataLoader(source_valid_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
    
    # 注意：few-shot 数据很少，batch_size 应当设小一点
    fs_batch_size = min(32, len(target_fewshot_dataset))
    target_fewshot_loader = DataLoader(target_fewshot_dataset, batch_size=fs_batch_size, shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)
    target_test_loader  = DataLoader(target_test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)

    global_num_classes = len(ALL_TERRAIN_CLASSES)
    model = VisionNet(num_classes=global_num_classes, head_type="wide").to(CONFIG["device"])
    criterion = nn.CrossEntropyLoss() 

    # =======================================================
    # STAGE 1: Base Training (在 Source 数据集上预训练)
    # =======================================================
    print(f"\n🚀 [阶段一] 开始 Base Training (Epochs: {CONFIG['base_epochs']}) ...")
    optimizer_base = optim.Adam(model.backbone.classifier.parameters(), lr=CONFIG["lr"])
    best_base_acc = 0.0
    patience = 5 
    counter = 0 
        
    for epoch in range(CONFIG["base_epochs"]):
        model.train()
        for inputs, labels in tqdm(source_train_loader, desc=f"Base Train Ep {epoch+1}"):
            inputs, labels = inputs.to(CONFIG["device"]), labels.to(CONFIG["device"])
            optimizer_base.zero_grad()
            outputs = model(inputs) 
            loss = criterion(outputs, labels) 
            loss.backward()
            optimizer_base.step()
            
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in source_valid_loader:
                inputs, labels = inputs.to(CONFIG["device"]), labels.to(CONFIG["device"])
                preds = torch.argmax(model(inputs), 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        print(f"   -> Base Val Acc: {val_acc:.4f}")

        if val_acc > best_base_acc + 0.001:
            best_base_acc = val_acc
            torch.save(model.state_dict(), best_base_model_path)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("   🛑 Base Training Early stopping triggered!")
                break

    # =======================================================
    # STAGE 2: Few-Shot Fine-Tuning (在 1% Target 数据上微调)
    # =======================================================
    print(f"\n🚀 [阶段二] 开始 Few-Shot 1% Fine-Tuning (Epochs: {CONFIG['fewshot_epochs']}) ...")
    
    # 重新加载最佳的 base model
    if os.path.exists(best_base_model_path):
        model.load_state_dict(torch.load(best_base_model_path, map_location=CONFIG['device']))
    
    # Fine-tuning 时可以稍微降低学习率
    optimizer_fs = optim.Adam(model.backbone.classifier.parameters(), lr=CONFIG["lr"] * 0.1)
    
    # 对于极少样本，通常不需要复杂的 valid 早停，直接跑到指定 Epoch 即可
    for epoch in range(CONFIG["fewshot_epochs"]):
        model.train()
        for inputs, labels in target_fewshot_loader:
            inputs, labels = inputs.to(CONFIG["device"]), labels.to(CONFIG["device"])
            optimizer_fs.zero_grad()
            outputs = model(inputs) 
            loss = criterion(outputs, labels) 
            loss.backward()
            optimizer_fs.step()
            
    # 保存微调后的模型
    torch.save(model.state_dict(), best_fs_model_path)

    # =======================================================
    # STAGE 3: Final Testing (在 Target Test 数据上测试)
    # =======================================================
    print("\n=== 🏆 训练结束，加载最佳 Few-Shot 模型进行最终盲测 ===")
    model.load_state_dict(torch.load(best_fs_model_path, map_location=CONFIG['device']))
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(target_test_loader, desc="Final Testing"):
            inputs, labels = inputs.to(CONFIG["device"]), labels.to(CONFIG["device"])
            preds = torch.argmax(model(inputs), 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    final_acc = accuracy_score(all_labels, all_preds)
    print(f"Final Few-Shot Test Accuracy: {final_acc:.4f}")

    unique_classes = sorted(list(set(all_labels)))
    target_names = [ALL_TERRAIN_CLASSES[i] for i in unique_classes]
    
    print(classification_report(all_labels, all_preds, labels=unique_classes, target_names=target_names, digits=4, zero_division=0))
    
    report_dict = classification_report(all_labels, all_preds, labels=unique_classes, target_names=target_names, digits=4, zero_division=0, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.loc['Final_Test_Accuracy'] = [final_acc, None, None, None]
    
    csv_filename = f"{args.exp_name}_fewshot_test_results.csv"
    csv_path = os.path.join(exp_save_dir, csv_filename)
    df_report.to_csv(csv_path, index=True)
    print(f"\n📊 [保存成功] 测试结果详细指标已导出至: {csv_path}")

if __name__ == '__main__':
    main(args)