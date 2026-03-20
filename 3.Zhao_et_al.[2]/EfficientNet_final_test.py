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
from torch.utils.data import Dataset, DataLoader, random_split # 新增：导入 random_split
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
        # S结构化城市道路
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/RSCD/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/ACDC/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/KITTI-360/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/IDD/processed_data/train/local_image_final"
        ],
        # U非结构化自然场景
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/DeepScene/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/TAS500/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/RELLIS/processed_data/test/local_image_final","/8TBHDD3/tht/TerraData/RELLIS/processed_data/valid/local_image_final"
        ]
    },
    "exp2": {
        # S结构化城市道路
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/RSCD/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/ACDC/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/KITTI-360/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/IDD/processed_data/train/local_image_final"
        ],
        # H半结构化混合场景
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/test",
            "/8TBHDD3/tht/TerraData/RUGD/processed_data/test/local_image_final","/8TBHDD3/tht/TerraData/RUGD/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/GOOSE/goose/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/GOOSE/gooseEx/processed_data/valid/local_image_final"
        ]
    },
    "exp3": {
            # U非结构化自然场景
            "train_dirs": [
                "/8TBHDD3/tht/TerraData/DeepScene/processed_data/train/local_image_final",
                "/8TBHDD3/tht/TerraData/WildScenes/processed_data/local_image_final",
                "/8TBHDD3/tht/TerraData/TAS500/processed_data/train/local_image_final",
                "/8TBHDD3/tht/TerraData/RELLIS/processed_data/train/local_image_final"
            ],
            
            # S结构化城市道路
            "test_dirs": [
                "/8TBHDD3/tht/TerraData/RSCD/processed_data/test/local_image_final",
                "/8TBHDD3/tht/TerraData/KITTI-360/processed_data/valid/local_image_final",
                "/8TBHDD3/tht/TerraData/IDD/processed_data/valid/local_image_final"
            ]
    },
    "exp4": {
        # U非结构化自然场景
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/DeepScene/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/WildScenes/processed_data/local_image_final",
            "/8TBHDD3/tht/TerraData/TAS500/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/RELLIS/processed_data/train/local_image_final"
        ],
        
        # H半结构化混合场景
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/test",
            "/8TBHDD3/tht/TerraData/RUGD/processed_data/test/local_image_final","/8TBHDD3/tht/TerraData/RUGD/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/GOOSE/goose/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/GOOSE/gooseEx/processed_data/valid/local_image_final"
        ]
    },
    "exp5": {
        # H半结构化混合场景
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/test",
            "/8TBHDD3/tht/TerraData/RUGD/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/GOOSE/goose/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/GOOSE/gooseEx/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/ORFD/processed_data/train/local_image_final"
        ],
        
        # S结构化城市道路
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/RSCD/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/KITTI-360/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/IDD/processed_data/valid/local_image_final"
        ]
    },
    "exp6": {
        # H半结构化混合场景
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/test",
            "/8TBHDD3/tht/TerraData/RUGD/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/GOOSE/goose/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/GOOSE/gooseEx/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/ORFD/processed_data/train/local_image_final"
        ],
        
        # U非结构化自然场景
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/DeepScene/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/TAS500/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/RELLIS/processed_data/test/local_image_final","/8TBHDD3/tht/TerraData/RELLIS/processed_data/valid/local_image_final"
        ]
    }
}




# ==========================================
# PART 1: 全局标签定义 (22类 全集) 
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
    "save_dir": "./efficientnet_checkpoints/benchmark2",

    # --- Zhao et al. (EfficientNet) 特有超参数 ---
    'lr_backbone': 0.0015,         # LR for CE loss
    'lr_center': 0.015,            # LR for Center loss
    'lambda_center': 2.0,          # Loss weight lambda
    'label_smoothing': 0.1,        # P_gamma = 0.9
    'lr_decay_gamma': 0.8          # Exponential decay
}

# ==========================================
# PART 3: 混合模式数据集类 (优化统计打印)
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
                self._load_folder_mode(folder_path)

        if len(self.samples) == 0:
            raise ValueError(f"❌ 未加载到 {self.dataset_type} 数据！请检查路径。")
            
        covered_classes = len(self.class_counts.keys())
        print(f"\n   📊 [{self.dataset_type}] 最终统计:")
        print(f"      - 总图片数: {len(self.samples)} 张")
        print(f"      - 覆盖类别: 全局 {len(ALL_TERRAIN_CLASSES)} 类中的 {covered_classes} 类。")
        
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
                label_str = item.get("material")
                if img_name and label_str:
                    label_str = label_str.lower()
                    if label_str in ALIAS_MAP: label_str = ALIAS_MAP[label_str]
                    if label_str in GLOBAL_LABEL_MAP:
                        full_img_path = os.path.join(folder_path, img_name)
                        if os.path.exists(full_img_path):
                            self.samples.append((full_img_path, label_str))
                            self.class_counts[label_str] += 1
                            valid_count += 1
            print(f"      -> {json_path.split('/')[-3]}: 成功加载 {valid_count} 张 (JSON模式)")
        except Exception as e:
            print(f"      -> [Error] 解析 {json_path} 失败: {e}")

    def _load_folder_mode(self, root_path):
        valid_count = 0
        for class_name in os.listdir(root_path):
            class_dir = os.path.join(root_path, class_name)
            if not os.path.isdir(class_dir): continue
                
            label_str = class_name.lower()
            if label_str in ALIAS_MAP: label_str = ALIAS_MAP[label_str]
            
            if label_str in GLOBAL_LABEL_MAP:
                files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                for img_file in files:
                    full_img_path = os.path.join(class_dir, img_file)
                    self.samples.append((full_img_path, label_str))
                    self.class_counts[label_str] += 1
                    valid_count += 1
        print(f"      -> {os.path.basename(root_path)}: 成功加载 {valid_count} 张 (文件夹模式)")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_str = self.samples[idx]
        label_idx = GLOBAL_LABEL_MAP[label_str]

        try:
            image = Image.open(img_path).convert('RGB')
        except:
            print(f"Warning: Corrupt image {img_path}")
            image = Image.new('RGB', (224, 224))
        
        if self.transform:
            image = self.transform(image)
        return image, label_idx


# ==========================================
# 新增: Dataset 包装器，用于给 Subset 分配不同的 Transform
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
# PART 5: 主程序 (动态路径与隔离解耦)
# ==========================================
def main(args):
    # 1. 提取实验配置并动态生成路径
    if args.exp_name not in EXPERIMENTS:
        raise ValueError(f"❌ 实验名称 '{args.exp_name}' 未在 EXPERIMENTS 中定义！")
    
    exp_cfg = EXPERIMENTS[args.exp_name]
    CONFIG["train_dirs"] = exp_cfg["train_dirs"]
    # 修复了旧代码中的读取bug，现读取 test_dirs
    CONFIG["test_dirs"] = exp_cfg["test_dirs"] 
    
    # 2. 动态创建此实验专属的保存目录，避免覆盖
    exp_save_dir = os.path.join(CONFIG["save_dir"], args.exp_name)
    os.makedirs(exp_save_dir, exist_ok=True)
    best_model_path = os.path.join(exp_save_dir, "efficientnet_hybrid_best.pth")

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

    # ------------------ 新的数据集划分逻辑 ------------------
    # 1. 加载包含全部训练数据的原始集 (暂不传入 transform)
    full_train_dataset = TerraHybridDataset(CONFIG["train_dirs"], CONFIG["json_name"], transform=None, is_train=True)
    
    # 2. 计算 80/20 切分比例
    total_size = len(full_train_dataset)
    train_size = int(0.8 * total_size)
    valid_size = total_size - train_size
    
    # 3. 随机切分
    raw_train_set, raw_valid_set = random_split(full_train_dataset, 
                                                [train_size, valid_size],
                                                generator=torch.Generator().manual_seed(42))
    
    # 4. 使用 Wrapper 给切分后的数据集分别挂载不同的 transform
    real_train_dataset = DatasetWrapper(raw_train_set, transform=train_transforms)
    real_valid_dataset = DatasetWrapper(raw_valid_set, transform=test_transforms)
    
    # 5. 加载真正的盲测数据 (Test Set)
    test_dataset = TerraHybridDataset(CONFIG["test_dirs"], CONFIG["json_name"], transform=test_transforms, is_train=False)

    print(f"\n📐 数据集划分完毕:")
    print(f"   - 训练集 (Train): {len(real_train_dataset)} 张")
    print(f"   - 验证集 (Valid): {len(real_valid_dataset)} 张 (用于早停和挑模型)")
    print(f"   - 测试集 (Test) : {len(test_dataset)} 张 (仅用于最终评价)")

    # 6. 生成三个 DataLoader
    train_loader = DataLoader(real_train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)
    valid_loader = DataLoader(real_valid_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
    # --------------------------------------------------------

    global_num_classes = len(ALL_TERRAIN_CLASSES)
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
        # --- Train ---
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

        # --- Valid ---
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

        # 保存最优模型
        if val_acc > best_acc + 0.001:  # patience_min_delta = 0.001
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f">>> 💾 Model Saved to {best_model_path}")
            
            counter = 0
            unique_classes = sorted(list(set(all_labels)))
            target_names = [ALL_TERRAIN_CLASSES[i] for i in unique_classes]
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
        # 🚨 重点：这里换成 test_loader 进行最终盲测！
        for images, labels in tqdm(test_loader, desc="Final Testing"):
            images, labels = images.to(CONFIG["device"]), labels.to(CONFIG["device"])
            
            _, logits = model(images)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    final_acc = accuracy_score(all_labels, all_preds)
    print(f"Final Test Accuracy: {final_acc:.4f}")

    unique_classes = sorted(list(set(all_labels)))
    target_names = [ALL_TERRAIN_CLASSES[i] for i in unique_classes]
    
    # 1. 打印到控制台看一眼
    print(classification_report(all_labels, all_preds, labels=unique_classes, target_names=target_names, digits=4, zero_division=0))
    
    # 2. 🌟 新增：将测试结果转换为 DataFrame 并保存为 CSV
    report_dict = classification_report(all_labels, all_preds, labels=unique_classes, target_names=target_names, digits=4, zero_division=0, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    
    # 👇 新加这一行！强行在 CSV 底部追加一行醒目的 Accuracy，专门方便你抄到论文表格里
    df_report.loc['Final_Test_Accuracy'] = [final_acc, None, None, None]
    
    csv_filename = f"{args.exp_name}_test_results.csv"
    csv_path = os.path.join(exp_save_dir, csv_filename)
    df_report.to_csv(csv_path, index=True)
    
    print(f"\n📊 [保存成功] 测试结果详细指标已导出至: {csv_path}")
    
    # ==============================================================================
    # 🚩 新增功能：汇总所有实验的 Final_Test_Accuracy 到总表
    # ==============================================================================
    
    # 1. 定义汇总表的保存路径
    summary_csv_path = os.path.join(CONFIG["save_dir"], "all_experiments_summary.csv")
    
    # 2. 确保总表所在的目录存在 (虽然理论上已存在，双重保险)
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    
    # 3. 定义所有实验的名称顺序 (按照 EXPERIMENTS 字典的定义顺序)
    exp_columns = list(EXPERIMENTS.keys())
    
    # 4. 读取或初始化汇总 DataFrame
    if os.path.exists(summary_csv_path):
        # 如果文件已存在，读取它
        df_summary = pd.read_csv(summary_csv_path)
        # 检查是否有新的 exp 列在文件中不存在，如果有则补上 (初始为 NaN)
        for col in exp_columns:
            if col not in df_summary.columns:
                df_summary[col] = float('nan')
        # 确保列的顺序和 EXPERIMENTS 定义的一致
        df_summary = df_summary[exp_columns]
    else:
        # 如果文件不存在，创建一个新的，只有一行，初始全为 NaN
        df_summary = pd.DataFrame(columns=exp_columns)
        df_summary.loc[0] = [float('nan')] * len(exp_columns)

    # 5. 更新当前实验的准确率
    # df_summary 只有一行 (index=0)，直接填入即可
    df_summary.at[0, args.exp_name] = final_acc

    # 6. 保存回 CSV
    df_summary.to_csv(summary_csv_path, index=False, float_format='%.6f')
    
    print(f"📈 [汇总更新] 已将 '{args.exp_name}' 结果录入总表: {summary_csv_path}")

if __name__ == '__main__':
    main(args)