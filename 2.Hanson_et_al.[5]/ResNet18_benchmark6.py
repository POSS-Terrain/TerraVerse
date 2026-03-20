import os
import argparse

# ==============================================================================
# 🚨 极其重要：必须在导入 torch 前完成参数解析和 GPU 环境变量设置！
# ==============================================================================
parser = argparse.ArgumentParser(description="多卡并发训练 ResNet18 Baseline")
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
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from collections import Counter
import pandas as pd

# ==============================================================================
# PART 0: 实验集配置 (EXPERIMENTS)
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

# ==========================================
# PART 1: 全局标签定义 [修改点：改为可通行性二分类]
# ==========================================
TRAVERSABILITY_CLASSES = ["non-traversable", "traversable"]
GLOBAL_LABEL_MAP = {label: idx for idx, label in enumerate(TRAVERSABILITY_CLASSES)}

# ==========================================
# PART 2: 配置参数 
# ==========================================
CONFIG = {
    "json_name": "local_label.json",
    "batch_size": 512,
    "lr": 1.5e-3,
    "epochs": 80, 
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 8,
    "img_size": 224,
    "save_dir": "./resnet18_checkpoints/benchmark6"
}

# ==========================================
# PART 3: 数据集类 [修改点：废弃文件夹模式，读取 traversability 字段]
# ==========================================
class TerraHybridDataset(Dataset):
    def __init__(self, dir_list, json_name, transform=None, is_train=False):
        self.transform = transform
        self.samples = [] 
        self.class_counts = Counter() 
        self.dataset_type = "Train/Valid (Raw)" if is_train else "Test"
        
        print(f"\n📂 [{self.dataset_type} Loader] 扫描 {len(dir_list)} 个路径...")
        
        for folder_path in dir_list:
            if not os.path.exists(folder_path):
                print(f"   ❌ [跳过] 路径不存在: {folder_path}")
                continue

            json_path = os.path.join(folder_path, json_name)
            if os.path.exists(json_path):
                self._load_json_mode(folder_path, json_path)
            else:
                # [修改点]：直接跳过无 JSON 文件的文件夹
                print(f"   ⚠️ [跳过] 未找到标签文件 {json_path}")

        if len(self.samples) == 0:
            raise ValueError(f"❌ 未加载到 {self.dataset_type} 数据！请检查路径。")
            
        covered_classes = len(self.class_counts.keys())
        print(f"\n   📊 [{self.dataset_type}] 最终统计:")
        print(f"      - 总图片数: {len(self.samples)} 张")
        print(f"      - 覆盖类别: 全局 {len(TRAVERSABILITY_CLASSES)} 类中的 {covered_classes} 类。")
        
        if covered_classes > 0:
            print(f"      - 类别明细 (降序):")
            for label, count in self.class_counts.most_common():
                print(f"        * {label}: {count} 张")

    def _load_json_mode(self, folder_path, json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            items = data.get("items", [])
            valid_count = 0
            for item in items:
                img_name = item.get("name")
                # [修改点]：提取 traversability
                trav_str = item.get("traversability")
                if img_name and trav_str:
                    trav_str = trav_str.lower()
                    if trav_str in GLOBAL_LABEL_MAP:
                        full_img_path = os.path.join(folder_path, img_name)
                        if os.path.exists(full_img_path):
                            self.samples.append((full_img_path, trav_str))
                            self.class_counts[trav_str] += 1
                            valid_count += 1
            print(f"      -> {json_path.split('/')[-3]}: 成功加载 {valid_count} 张 (JSON模式)")
        except Exception as e:
            print(f"      -> [Error] JSON Fail: {e}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, trav_str = self.samples[idx]
        label_idx = GLOBAL_LABEL_MAP[trav_str]

        try:
            image = Image.open(img_path).convert('RGB')
        except:
            print(f"Warning: Corrupt image {img_path}")
            image = Image.new('RGB', (224, 224))
        
        if self.transform:
            image = self.transform(image)
        return image, label_idx

# ==========================================
# Dataset 包装器
# ==========================================
class DatasetWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.subset[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.subset)

# ==========================================
# PART 4: ResNet18 模型 (保持原样)
# ==========================================
class VASTVisionBaseline(nn.Module):
    def __init__(self, num_classes=7):
        super(VASTVisionBaseline, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# ==========================================
# PART 5: 主程序
# ==========================================
def main(args):
    if args.exp_name not in EXPERIMENTS:
        raise ValueError(f"❌ 实验名称 '{args.exp_name}' 未在 EXPERIMENTS 中定义！")
    
    exp_cfg = EXPERIMENTS[args.exp_name]
    CONFIG["train_dirs"] = exp_cfg["train_dirs"]
    CONFIG["test_dirs"] = exp_cfg["test_dirs"] 
    
    exp_save_dir = os.path.join(CONFIG["save_dir"], args.exp_name)
    os.makedirs(exp_save_dir, exist_ok=True)
    # [修改点]：更换权重名称
    best_model_path = os.path.join(exp_save_dir, "resnet18_traversable_best.pth")

    print(f"\n=======================================================")
    print(f"🚀 启动 ResNet18 实验: {args.exp_name} | 使用设备: GPU {args.gpu}")
    print(f"📁 模型保存至: {exp_save_dir}")
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

    full_train_dataset = TerraHybridDataset(CONFIG["train_dirs"], CONFIG["json_name"], transform=None, is_train=True)
    
    total_size = len(full_train_dataset)
    train_size = int(0.8 * total_size)
    valid_size = total_size - train_size
    
    raw_train_set, raw_valid_set = random_split(full_train_dataset, 
                                                [train_size, valid_size],
                                                generator=torch.Generator().manual_seed(42))
    
    real_train_dataset = DatasetWrapper(raw_train_set, transform=train_transforms)
    real_valid_dataset = DatasetWrapper(raw_valid_set, transform=test_transforms)
    
    test_dataset = TerraHybridDataset(CONFIG["test_dirs"], CONFIG["json_name"], transform=test_transforms, is_train=False)

    print(f"\n📐 数据集划分完毕:")
    print(f"   - 训练集 (Train): {len(real_train_dataset)} 张")
    print(f"   - 验证集 (Valid): {len(real_valid_dataset)} 张 (用于早停和挑模型)")
    print(f"   - 测试集 (Test) : {len(test_dataset)} 张 (仅用于最终评价)")

    train_loader = DataLoader(real_train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)
    valid_loader = DataLoader(real_valid_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)

    # [修改点]：应用二分类
    global_num_classes = len(TRAVERSABILITY_CLASSES)
    print(f"\n🏗️ 初始化模型: Output Classes = {global_num_classes}")
    
    model = VASTVisionBaseline(num_classes=global_num_classes).to(CONFIG["device"])
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss() 

    print(f"\n🚀 开始训练 (Epochs: {CONFIG['epochs']}) ...")
    best_acc = 0.0
    patience = 5 
    counter = 0 
        
    for epoch in range(CONFIG["epochs"]):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(CONFIG["device"]), labels.to(CONFIG["device"])
            
            optimizer.zero_grad()
            outputs = model(inputs) 
            loss = criterion(outputs, labels) 
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in tqdm(valid_loader, desc="Validating"):
                inputs, labels = inputs.to(CONFIG["device"]), labels.to(CONFIG["device"])
                outputs = model(inputs)
                preds = torch.argmax(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Result | Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc + 0.001:  
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f">>> 💾 Model Saved to {best_model_path}")
            
            counter = 0
            unique_classes = sorted(list(set(all_labels)))
            # [修改点]：在验证阶段也要映射二分类的名称
            target_names = [TRAVERSABILITY_CLASSES[i] for i in unique_classes]
            print(classification_report(all_labels, all_preds, labels=unique_classes, target_names=target_names, digits=4, zero_division=0))
            
        else:
            counter += 1 
            print(f"EarlyStopping counter: {counter} out of {patience}")
            
        if counter >= patience:
            print("🛑 Early stopping triggered!")
            break

    print("\n=== 🏆 训练结束，加载最佳模型进行最终盲测 (Test) ===")

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=CONFIG['device']))
        print(f"✅ 已成功加载最佳模型: {best_model_path}")
    else:
        print("⚠️ 警告: 未找到最佳模型，将使用当前 Epoch 权重进行测试。")

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Final Testing"):
            inputs, labels = inputs.to(CONFIG["device"]), labels.to(CONFIG["device"])
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    final_acc = accuracy_score(all_labels, all_preds)
    print(f"Final Test Accuracy: {final_acc:.4f}")

    unique_classes = sorted(list(set(all_labels)))
    # [修改点]：测试评价阶段名称映射
    target_names = [TRAVERSABILITY_CLASSES[i] for i in unique_classes]
    
    print(classification_report(all_labels, all_preds, labels=unique_classes, target_names=target_names, digits=4, zero_division=0))
    
    report_dict = classification_report(all_labels, all_preds, labels=unique_classes, target_names=target_names, digits=4, zero_division=0, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    
    df_report.loc['Final_Test_Accuracy'] = [final_acc, None, None, None]
    
    # [修改点]：更新 CSV 文件名
    csv_filename = f"{args.exp_name}_traversable_test_results.csv"
    csv_path = os.path.join(exp_save_dir, csv_filename)
    df_report.to_csv(csv_path, index=True)
    
    print(f"\n📊 [保存成功] 测试结果详细指标已导出至: {csv_path}")

if __name__ == '__main__':
    main(args)