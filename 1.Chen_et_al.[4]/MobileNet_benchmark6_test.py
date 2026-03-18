import os
import argparse

# ==============================================================================
# 🚨 极其重要：必须在导入 torch 前完成参数解析和 GPU 环境变量设置！
# ==============================================================================
parser = argparse.ArgumentParser(description="MobileNetV2 纯测试脚本")
parser.add_argument("--exp_name", type=str, required=True, help="实验名称，如 exp1, exp2，用于匹配测试集路径")
parser.add_argument("--gpu", type=str, default="0", help="指定要使用的 GPU ID，如 0 或 1")
# [修改点]：这里更新为了正确的二分类模型权重路径！
parser.add_argument("--ckpt_path", type=str, 
                    default="/8TBHDD3/tht/TerraX/6Chen_et_al./mobilenet_checkpoints/benchmark1/exp2/mobilenet_traversable_best.pth", 
                    help="训练好的模型权重路径")
args, _ = parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# ==============================================================================
# --- 环境设置完毕后，再导入深度学习包 ---
# ==============================================================================
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from collections import Counter
import pandas as pd

# ==============================================================================
# PART 0: 实验集配置 (EXPERIMENTS) - 仅保留 test_dirs
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

# ==========================================
# PART 1: 全局标签定义 & 配置参数
# ==========================================
TRAVERSABILITY_CLASSES = ["non-traversable", "traversable"]
GLOBAL_LABEL_MAP = {label: idx for idx, label in enumerate(TRAVERSABILITY_CLASSES)}

CONFIG = {
    "json_name": "local_label.json",
    "batch_size": 512,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 8,
    "img_size": 224
}

# ==========================================
# PART 2: 数据集类
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
            image = Image.new('RGB', (224, 224))
        
        if self.transform:
            image = self.transform(image)
        return image, label_idx

# ==========================================
# PART 3: MobileNetV2 模型
# ==========================================
class VisionNet(nn.Module):
    def __init__(self, num_classes, head_type="wide"):
        super(VisionNet, self).__init__()
        self.backbone = models.mobilenet_v2(weights=None)
        
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
# PART 4: 主程序 (纯测试逻辑)
# ==========================================
def main(args):
    # 1. 提取实验配置
    if args.exp_name not in EXPERIMENTS:
        raise ValueError(f"❌ 实验名称 '{args.exp_name}' 未在 EXPERIMENTS 中定义！")
    
    CONFIG["test_dirs"] = EXPERIMENTS[args.exp_name]["test_dirs"]

    print(f"\n=======================================================")
    print(f"🚀 启动 MobileNet 测试: {args.exp_name} | 使用设备: GPU {args.gpu}")
    print(f"📦 目标模型路径: {args.ckpt_path}")
    print(f"=======================================================\n")
    
    # 定义数据预处理 (仅需测试集处理)
    test_transforms = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. 加载测试集
    test_dataset = TerraHybridDataset(CONFIG["test_dirs"], CONFIG["json_name"], transform=test_transforms, is_train=False)
    test_loader  = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)

    # 3. 初始化模型并加载权重
    global_num_classes = len(TRAVERSABILITY_CLASSES)
    model = VisionNet(num_classes=global_num_classes, head_type="wide").to(CONFIG["device"])
    
    if os.path.exists(args.ckpt_path):
        model.load_state_dict(torch.load(args.ckpt_path, map_location=CONFIG['device']))
        print(f"\n✅ 已成功加载模型权重: {args.ckpt_path}")
    else:
        raise FileNotFoundError(f"❌ 未找到模型权重文件: {args.ckpt_path}，请检查路径！")
        
    # 4. 运行盲测
    print("\n=== 🏆 开始进行盲测 (Testing) ===")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(CONFIG["device"]), labels.to(CONFIG["device"])
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 5. 计算指标并打印
    final_acc = accuracy_score(all_labels, all_preds)
    print(f"\nFinal Test Accuracy: {final_acc:.4f}")

    unique_classes = sorted(list(set(all_labels)))
    target_names = [TRAVERSABILITY_CLASSES[i] for i in unique_classes]
    
    print("\n" + classification_report(all_labels, all_preds, labels=unique_classes, target_names=target_names, digits=4, zero_division=0))
    
    # 6. 保存报告到当前目录
    report_dict = classification_report(all_labels, all_preds, labels=unique_classes, target_names=target_names, digits=4, zero_division=0, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.loc['Final_Test_Accuracy'] = [final_acc, None, None, None]
    
    csv_filename = f"{args.exp_name}_inference_results.csv"
    df_report.to_csv(csv_filename, index=True)
    
    print(f"\n📊 [保存成功] 测试结果详细指标已导出至当前目录: ./{csv_filename}")

if __name__ == '__main__':
    main(args)