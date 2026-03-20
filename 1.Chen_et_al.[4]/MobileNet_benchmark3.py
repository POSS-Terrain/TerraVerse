import os
import argparse

# ==============================================================================
# 🚨 极其重要：必须在导入 torch 前完成参数解析和 GPU 环境变量设置！
# ==============================================================================
parser = argparse.ArgumentParser(description="多卡并发训练 MobileNetV2 (Benchmark 3)")
parser.add_argument("--exp_name", type=str, required=True, help="实验名称，此处建议填 benchmark3")
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
import csv

# ==============================================================================
# PART 0: 实验集配置 (EXPERIMENTS) - 针对 Benchmark 3 重构
# ==============================================================================
EXPERIMENTS = {
    "benchmark3": {
        # 统一的训练集 (Normal)： TerraPOSS, Jackal
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/train/local_image_final"
        ],
        # 🏆 一次性定义 5 种不同天气的测试套件 (Test Suites)
        "test_suites": {
            "Normal": [
                "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/test/local_image_final",
                "/8TBHDD3/tht/TerraData/Jackal/processed_data/test/local_image_final"
            ],
            "Dark": [
                "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/dark/local_image_final",
                "/8TBHDD3/tht/TerraData/Jackal/processed_data/dark/local_image_final"
            ],
            "Foggy": [
                "/8TBHDD3/tht/TerraData/Jackal/processed_data/sim_fog/local_image_final"
            ],
            "Snowy": [
                "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/snow/local_image_final"
            ],
            "Sunny": [
                "/8TBHDD3/tht/TerraData/Jackal/processed_data/sim_sun/local_image_final"
            ]
        }
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
    "epochs": 80, 
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 8,
    "img_size": 224,
    "save_dir": "./mobilenet_checkpoints" 
}

# ==========================================
# PART 2: 混合模式数据集类 & Wrapper 
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

    def _load_json_mode(self, folder_path, json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            items = data.get("items", [])
            valid_count = 0
            for item in items:
                img_name = item.get("name")
                label_str = item.get("label", "").lower()
                if img_name and label_str:
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
                files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
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
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.subset)

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

    def forward(self, x):
        return self.backbone(x)

# ==========================================
# PART 4: 主程序
# ==========================================
def main(args):
    if args.exp_name not in EXPERIMENTS:
        raise ValueError(f"❌ 实验名称 '{args.exp_name}' 未在 EXPERIMENTS 中定义！试试填 benchmark3")
    
    exp_cfg = EXPERIMENTS[args.exp_name]
    CONFIG["train_dirs"] = exp_cfg["train_dirs"]
    CONFIG["test_suites"] = exp_cfg["test_suites"] # 提取多测试集套件
    
    exp_save_dir = os.path.join(CONFIG["save_dir"], args.exp_name)
    os.makedirs(exp_save_dir, exist_ok=True)
    best_model_path = os.path.join(exp_save_dir, "mobilenet_hybrid_best.pth")

    print(f"\n=======================================================")
    print(f"🚀 启动 Benchmark 实验: {args.exp_name} | GPU: {args.gpu}")
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

    # ------------------ 数据集划分 ------------------
    full_train_dataset = TerraHybridDataset(CONFIG["train_dirs"], CONFIG["json_name"], transform=None, is_train=True)
    
    train_size = int(0.8 * len(full_train_dataset))
    valid_size = len(full_train_dataset) - train_size
    raw_train_set, raw_valid_set = random_split(full_train_dataset, [train_size, valid_size])
    
    real_train_dataset = DatasetWrapper(raw_train_set, transform=train_transforms)
    real_valid_dataset = DatasetWrapper(raw_valid_set, transform=test_transforms)

    train_loader = DataLoader(real_train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)
    valid_loader = DataLoader(real_valid_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
    # --------------------------------------------------------

    global_num_classes = len(ALL_TERRAIN_CLASSES)
    model = VisionNet(num_classes=global_num_classes, head_type="wide").to(CONFIG["device"])
    
    optimizer = optim.Adam(model.backbone.classifier.parameters(), lr=CONFIG["lr"])
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
        all_preds, all_labels = [], []
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
            counter = 0
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter} out of {patience}")
            
        if counter >= patience:
            print("🛑 Early stopping triggered!")
            break

    # =========================================================================
    # 🌟 核心修改：统一测试环节 (Multiple Test Suites Evaluation)
    # =========================================================================
    print("\n=== 🏆 训练结束，加载最佳模型进行全环境测试 (Benchmark 3) ===")

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=CONFIG['device']))
        print(f"✅ 已成功加载最佳模型: {best_model_path}")
    else:
        print("⚠️ 警告: 未找到最佳模型，使用当前 Epoch 权重进行测试。")
        
    model.eval()
    
    # 存储所有天气的最终成绩，方便最后汇总打印
    benchmark_results = {}

    with torch.no_grad():
        # 遍历 5 种不同的天气测试集
        for condition_name, test_dirs in CONFIG["test_suites"].items():
            print(f"\n[Testing] 正在测试环境: {condition_name}")
            
            # 动态加载当前天气的 Dataset 和 DataLoader
            test_dataset = TerraHybridDataset(test_dirs, CONFIG["json_name"], transform=test_transforms, is_train=False)
            test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
            
            all_preds, all_labels = [], []
            for inputs, labels in tqdm(test_loader, desc=f"Eval {condition_name}"):
                inputs, labels = inputs.to(CONFIG["device"]), labels.to(CONFIG["device"])
                outputs = model(inputs)
                preds = torch.argmax(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            final_acc = accuracy_score(all_labels, all_preds)
            benchmark_results[condition_name] = final_acc
            print(f">>> {condition_name} Test Accuracy: {final_acc:.4f}")
            
            # 打印该环境下的详细分类报告
            unique_classes = sorted(list(set(all_labels)))
            target_names = [ALL_TERRAIN_CLASSES[i] for i in unique_classes]
            print(classification_report(all_labels, all_preds, labels=unique_classes, target_names=target_names, digits=4, zero_division=0))

    # =========================================================================
    # 打印并保存最终漂亮的汇总表格数据
    # =========================================================================
    print("\n=======================================================")
    print("📊 Benchmark 3: Robust Classification Final Results")
    print("=======================================================")
    
    # 1. 终端打印
    for condition, acc in benchmark_results.items():
        print(f"| {condition.ljust(10)} | {acc:.4f} |")
    print("=======================================================\n")

    # 2. 自动保存为 CSV 文件 (方便 Excel 打开和复制粘贴)
    results_csv_path = os.path.join(exp_save_dir, f"{args.exp_name}_results.csv")
    
    with open(results_csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(["Test Condition", "Accuracy"])
        # 写入数据
        for condition, acc in benchmark_results.items():
            writer.writerow([condition, f"{acc:.4f}"])
            
    print(f"📁 汇总结果已成功保存至: {results_csv_path}")

if __name__ == '__main__':
    main(args)