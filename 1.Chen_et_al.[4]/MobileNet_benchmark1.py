import os
import argparse

# ==============================================================================
# 🚨 极其重要：必须在导入 torch 前完成参数解析和 GPU 环境变量设置！
# ==============================================================================
parser = argparse.ArgumentParser(description="多卡并发训练 MobileNetV2")
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
# PART 0: 实验集配置 (EXPERIMENTS) - 保持你修改好的不变
# ==============================================================================
EXPERIMENTS = {
    "exp1": {
        # COCO-Stuff
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/COCO-Stuff/processed_data/train/local_image_final"
        ],
       
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/COCO-Stuff/processed_data/valid/local_image_final"
        ]
    },
    "exp2": {
        # DeepScene
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/DeepScene/processed_data/train/local_image_final"
        ],
        
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/DeepScene/processed_data/test/local_image_final"
        ]
    },
    "exp3": {
        # FCDD
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/FCDD/processed_data/train/local_image_final"
        ],
         
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/FCDD/processed_data/valid/local_image_final"
        ]
    },
    "exp4": {
        # GOOSE
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/GOOSE/goose/processed_data/train/local_image_final"
        ],
        
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/GOOSE/goose/processed_data/valid/local_image_final"
        ]
    },
    "exp5": {
        # GOOSE-ex
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/GOOSE/gooseEx/processed_data/train/local_image_final"
        ],
        
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/GOOSE/gooseEx/processed_data/valid/local_image_final"
        ]
    },
    "exp6": {
        # Jackal
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/train/local_image_final"
        ],
        
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/test/local_image_final"
        ]
    },
    "exp7": {
        # KITTI-360
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/KITTI-360/processed_data/train/local_image_final"
        ],
        
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/KITTI-360/processed_data/valid/local_image_final"
        ]
    },
    "exp8": {
        # RELLIS
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/RELLIS/processed_data/train/local_image_final"
        ],
        
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/RELLIS/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/RELLIS/processed_data/valid/local_image_final"
        ]
    },
    "exp9": {
        # RSCD
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/RSCD/processed_data/train/local_image_final"
        ],
        
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/RSCD/processed_data/test/local_image_final"
        ]
    },
    "exp10": {
        # RUGD
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/RUGD/processed_data/train/local_image_final"
        ],
        
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/RUGD/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/RUGD/processed_data/valid/local_image_final"
        ]
    },
    "exp11": {
        # TAS500
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/TAS500/processed_data/train/local_image_final"
        ],
        
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/TAS500/processed_data/valid/local_image_final"
        ]
    },
    "exp12": {
        # TerraPOSS
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/train/local_image_final"
        ],
        
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/test/local_image_final"
        ]
    },
    "exp13": {
        # VAST
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/train"
        ],
        
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/test"
        ]
    },
    "exp14": {
        # YCOR
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/YCOR/processed_data/train/local_image_final"
        ],
        
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/YCOR/processed_data/valid/local_image_final"
        ]
    },
    "exp15": {
        # ORAD-3D-Label
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/ORAD-3D-Label/processed_data/train/local_image_final"
        ],
        
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/ORAD-3D-Label/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/ORAD-3D-Label/processed_data/valid/local_image_final"
        ]
    },
    "exp16": {
        # test_dirs不加入全集：  IDD
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/IDD/processed_data/train/local_image_final"
        ],
        
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/IDD/processed_data/valid/local_image_final"
        ]
    },
    "exp17": {
        # test_dirs不加入全集：  ACDC
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/ACDC/processed_data/train/local_image_final"
        ],
        
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/ACDC/processed_data/valid/local_image_final"
        ]
    },
    "exp18": {
        # test_dirs不加入全集：  ORFD
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/ORFD/processed_data/train/local_image_final"
        ],
        
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/ORFD/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/ORFD/processed_data/valid/local_image_final"
        ]
    },
    "exp19": {
        # test_dirs不加入全集：  RTK
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/RTK/processed_data/local_image_final"
        ],
        
        "test_dirs": [
            
        ]
    },
    "exp20": {
        # test_dirs不加入全集：  WildScenes
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/WildScenes/processed_data/local_image_final"
        ],
        
        "test_dirs": [
            
        ]
    },
    "exp21": {
        # 综上所述的全集
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/COCO-Stuff/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/DeepScene/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/FCDD/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/GOOSE/goose/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/GOOSE/gooseEx/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/KITTI-360/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/RELLIS/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/RSCD/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/RUGD/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/TAS500/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/train",
            "/8TBHDD3/tht/TerraData/YCOR/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/ORAD-3D-Label/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/IDD/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/ACDC/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/ORFD/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/RTK/processed_data/local_image_final",
            "/8TBHDD3/tht/TerraData/WildScenes/processed_data/local_image_final"
        ],
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/COCO-Stuff/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/DeepScene/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/FCDD/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/GOOSE/goose/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/GOOSE/gooseEx/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/RELLIS/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/RELLIS/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/RSCD/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/RUGD/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/RUGD/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/TAS500/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/test",
            "/8TBHDD3/tht/TerraData/YCOR/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/ORAD-3D-Label/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/ORAD-3D-Label/processed_data/valid/local_image_final"
        ]
    }
}


# ==========================================
# PART 1: 全局标签定义 & 配置参数 (保持不变)
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
    "batch_size": 512,
    "lr": 1.5e-3,
    "epochs": 80, 
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 8,
    "img_size": 224,
    "save_dir": "./mobilenet_checkpoints/benchmark1" 
}

# ==========================================
# PART 2: 混合模式数据集类 
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
            print(f"      -> [Error] JSON Fail: {e}")

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
# PART 3: MobileNetV2 模型 (保持不变)
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
# PART 3.5: 实用评估辅助函数 (新增)
# ==========================================
def group_test_dirs(test_dirs):
    """
    根据路径自动将测试集归类。
    """
    from collections import defaultdict
    grouped = defaultdict(list)
    for path in test_dirs:
        if "TerraData/" in path:
            dataset_name = path.split("TerraData/")[1].split("/")[0]
            if dataset_name == "GOOSE":
                dataset_name = "GOOSE_" + path.split("GOOSE/")[1].split("/")[0]
        else:
            dataset_name = "Unknown_" + path.split("/")[-3]
        grouped[dataset_name].append(path)
    return grouped

def evaluate_mobilenet(model, loader, device):
    """
    专门为 MobileNet 抽取的评估函数
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    return acc, all_labels, all_preds

# ==========================================
# PART 4: 主程序 (核心修改区域)
# ==========================================
def main(args):
    # 1. 提取实验配置
    if args.exp_name not in EXPERIMENTS:
        raise ValueError(f"❌ 实验名称 '{args.exp_name}' 未在 EXPERIMENTS 中定义！")
    
    exp_cfg = EXPERIMENTS[args.exp_name]
    CONFIG["train_dirs"] = exp_cfg["train_dirs"]
    CONFIG["test_dirs"] = exp_cfg["test_dirs"] # 注意这里改为了 test_dirs
    
    # 2. 动态创建此实验专属的保存目录
    exp_save_dir = os.path.join(CONFIG["save_dir"], args.exp_name)
    os.makedirs(exp_save_dir, exist_ok=True)
    best_model_path = os.path.join(exp_save_dir, "mobilenet_hybrid_best.pth")

    print(f"\n=======================================================")
    print(f"🚀 启动 MobileNet 实验: {args.exp_name} | 使用设备: GPU {args.gpu}")
    print(f"📁 模型保存至: {exp_save_dir}")
    print(f"=======================================================\n")
    
    # 定义数据增强
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

    # ------------------ 新的数据集划分逻辑 ------------------
    # 1. 加载包含全部训练数据的原始集 (暂不传入 transform，获取 PIL 图像)
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
        all_preds = []
        all_labels = []
        with torch.no_grad():
            # 这里继续使用 valid_loader 看每轮的效果
            for inputs, labels in tqdm(valid_loader, desc="Validating"):
                inputs, labels = inputs.to(CONFIG["device"]), labels.to(CONFIG["device"])
                outputs = model(inputs)
                preds = torch.argmax(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Result | Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc + 0.001:  # patience_min_delta = 0.001
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f">>> 💾 Model Saved to {best_model_path}")
            
            counter = 0
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
        
    # ---------------------------------------------------------
    # 1. 首先进行【总体】测试集评估 (All Mixed Test)
    # ---------------------------------------------------------
    print("\n>>> 正在评估: 总体测试集 (Overall Test Set)")
    final_acc, all_labels, all_preds = evaluate_mobilenet(model, test_loader, CONFIG["device"])
    print(f"Final Overall Test Accuracy: {final_acc:.4f}")

    unique_classes = sorted(list(set(all_labels)))
    target_names = [ALL_TERRAIN_CLASSES[i] for i in unique_classes]
    
    # 控制台打印总体报告
    print(classification_report(all_labels, all_preds, labels=unique_classes, target_names=target_names, digits=4, zero_division=0))
    
    # 导出总体结果 CSV
    report_dict = classification_report(all_labels, all_preds, labels=unique_classes, target_names=target_names, digits=4, zero_division=0, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.loc['Final_Test_Accuracy'] = [final_acc, None, None, None]
    
    overall_csv_path = os.path.join(exp_save_dir, f"{args.exp_name}_overall_test_results.csv")
    df_report.to_csv(overall_csv_path, index=True)
    print(f"📊 [保存成功] 总体测试结果已导出至: {overall_csv_path}")

    # ---------------------------------------------------------
    # 2. 准备进行【子集】测试集单独评估
    # ---------------------------------------------------------
    grouped_test_dirs = group_test_dirs(CONFIG["test_dirs"])
    
    # 创建子集报告专属文件夹
    subset_reports_dir = os.path.join(exp_save_dir, "subset_reports")
    os.makedirs(subset_reports_dir, exist_ok=True)
    
    # 核心数据结构：用于生成汇总对比表
    summary_results = {
        "Dataset": ["Overall_All_Mixed"],
        "Accuracy": [final_acc],
        "Sample_Count": [len(test_dataset)]
    }

    for subset_name, dir_list in grouped_test_dirs.items():
        print(f"\n>>> 正在评估子数据集: {subset_name} ({len(dir_list)} 个路径合并)")
        
        # 针对当前子集实例化 Dataset
        subset_dataset = TerraHybridDataset(dir_list, CONFIG["json_name"], transform=test_transforms, is_train=False)
        
        if len(subset_dataset) == 0:
            print(f"⚠️ 子集 {subset_name} 为空，跳过评估。")
            continue
            
        subset_loader = DataLoader(subset_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
        
        # 评估该子集
        sub_acc, sub_targets, sub_preds = evaluate_mobilenet(model, subset_loader, CONFIG["device"])
        print(f"[{subset_name}] Accuracy: {sub_acc:.4f}")
        
        # 记录到汇总表
        summary_results["Dataset"].append(subset_name)
        summary_results["Accuracy"].append(sub_acc)
        summary_results["Sample_Count"].append(len(subset_dataset))
        
        # 保存该子集的详细分类报告 CSV
        sub_unique_classes = sorted(list(set(sub_targets)))
        sub_target_names = [ALL_TERRAIN_CLASSES[i] for i in sub_unique_classes]
        
        sub_report_dict = classification_report(sub_targets, sub_preds, labels=sub_unique_classes, target_names=sub_target_names, digits=4, zero_division=0, output_dict=True)
        sub_df = pd.DataFrame(sub_report_dict).transpose()
        sub_df.loc['Subset_Accuracy'] = [sub_acc, None, None, None]
        
        sub_csv_path = os.path.join(subset_reports_dir, f"{subset_name}_results.csv")
        sub_df.to_csv(sub_csv_path, index=True)

    # ---------------------------------------------------------
    # 3. 导出终极的【子数据集精度汇总表】
    # ---------------------------------------------------------
    df_summary = pd.DataFrame(summary_results)
    
    # 保留第一行 Overall，剩余的按照 Accuracy 降序排列，方便直观查看薄弱场景
    overall_row = df_summary.iloc[[0]]
    subsets_sorted = df_summary.iloc[1:].sort_values(by="Accuracy", ascending=False)
    df_summary_final = pd.concat([overall_row, subsets_sorted]).reset_index(drop=True)
    
    summary_csv_path = os.path.join(exp_save_dir, f"{args.exp_name}_subsets_accuracy_summary.csv")
    df_summary_final.to_csv(summary_csv_path, index=False)
    
    print("\n=======================================================")
    print(f"🎯 所有评估已顺利完成！")
    print(f"📄 总体详细报告 (Overall): {overall_csv_path}")
    print(f"📊 各子集准确率汇总对比表: {summary_csv_path}")
    print(f"📁 各子集详细分类报告目录: {subset_reports_dir}/")
    print("=======================================================")

if __name__ == '__main__':
    main(args)