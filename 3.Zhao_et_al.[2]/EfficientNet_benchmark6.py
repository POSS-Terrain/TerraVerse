import os
import argparse

# ==============================================================================
# 🚨 极其重要：必须在导入 torch 前完成参数解析和 GPU 环境变量设置！
# ==============================================================================
parser = argparse.ArgumentParser(description="多卡并发训练 EfficientNet (Center Loss)")
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
    "epochs": 80, 
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 8,
    "img_size": 224,
    "save_dir": "./efficientnet_checkpoints/benchmark6",

    # --- Zhao et al. (EfficientNet) 特有超参数 ---
    'lr_backbone': 0.0015,         # LR for CE loss
    'lr_center': 0.015,            # LR for Center loss
    'lambda_center': 2.0,          # Loss weight lambda
    'label_smoothing': 0.1,        # P_gamma = 0.9
    'lr_decay_gamma': 0.8          # Exponential decay
}

# ==========================================
# PART 3: 混合模式数据集类 [修改点：废弃文件夹模式，读取 traversability 字段]
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
                # [修改点]：跳过无 JSON 文件的文件夹
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
            print(f"      -> [Error] 解析 {json_path} 失败: {e}")

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
# PART 4: EfficientNet 模型与 CenterLoss (保持原样)
# ==========================================
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.to(x.device)
        
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss * 0.5

class ZhaoVisionModel(nn.Module):
    def __init__(self, num_classes=7):
        super(ZhaoVisionModel, self).__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity() 
        self.fc = nn.Linear(in_features, num_classes)
        self.feature_dim = in_features

    def forward(self, x):
        embedding = self.backbone(x)
        logits = self.fc(embedding)
        return embedding, logits

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
    # [修改点]：更新保存权重的文件名
    best_model_path = os.path.join(exp_save_dir, "efficientnet_traversable_best.pth")

    print(f"\n=======================================================")
    print(f"🚀 启动 EfficientNet 实验: {args.exp_name} | 使用设备: GPU {args.gpu}")
    print(f"📁 模型保存至: {exp_save_dir}")
    print(f"=======================================================\n")
    
    train_transforms = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])), 
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.1),
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
    
    model = ZhaoVisionModel(num_classes=global_num_classes).to(CONFIG["device"])
    
    feat_dim = model.feature_dim 
    center_loss = CenterLoss(num_classes=global_num_classes, feat_dim=feat_dim, use_gpu=(CONFIG['device']=='cuda')).to(CONFIG['device'])

    optimizer_model = optim.Adam(model.parameters(), lr=CONFIG['lr_backbone'])
    optimizer_center = optim.Adam(center_loss.parameters(), lr=CONFIG['lr_center'])
    
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer_model, gamma=CONFIG['lr_decay_gamma'])
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])

    print(f"\n🚀 开始训练 (Epochs: {CONFIG['epochs']}) ...")
    print(f"   Config: EffNet-B0 | Lambda Center={CONFIG['lambda_center']} | Label Smoothing={CONFIG['label_smoothing']}")
    
    best_acc = 0.0
    patience = 5 
    counter = 0 
        
    for epoch in range(CONFIG["epochs"]):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for images, labels in pbar:
            images, labels = images.to(CONFIG["device"]), labels.to(CONFIG["device"])
            
            embeddings, logits = model(images)
            
            loss_ce = criterion_ce(logits, labels)
            loss_center_val = center_loss(embeddings, labels)
            loss = loss_ce + CONFIG['lambda_center'] * loss_center_val
            
            optimizer_model.zero_grad()
            optimizer_center.zero_grad()
            
            loss.backward()
            
            optimizer_model.step()
            for param in center_loss.parameters():
                param.grad.data *= (1. / CONFIG['lambda_center']) 
            optimizer_center.step()
            
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), ce=loss_ce.item())

        scheduler.step()

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in tqdm(valid_loader, desc="Validating"):
                images, labels = images.to(CONFIG["device"]), labels.to(CONFIG["device"])
                
                _, logits = model(images)
                preds = torch.argmax(logits, dim=1)
                
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
            # [修改点]：映射分类名称
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
        print("⚠️ 警告: 未找到最佳模型，使用当前 Epoch 权重进行测试。")
        
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Final Testing"):
            images, labels = images.to(CONFIG["device"]), labels.to(CONFIG["device"])
            
            _, logits = model(images)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    final_acc = accuracy_score(all_labels, all_preds)
    print(f"Final Test Accuracy: {final_acc:.4f}")

    unique_classes = sorted(list(set(all_labels)))
    # [修改点]：映射分类名称
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