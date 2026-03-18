import os
import argparse

# ==============================================================================
# 🚨 极其重要：必须在导入 torch 前完成参数解析和 GPU 环境变量设置！
# ==============================================================================
parser = argparse.ArgumentParser(description="MobileNetV2 纯测试脚本 (支持单实验或批量遍历)")
parser.add_argument("--exp_name", type=str, required=True, 
                    help="实验名称，如 exp1, exp2 ... exp6。输入 'all' 则自动遍历执行所有实验")
parser.add_argument("--gpu", type=str, default="0", help="指定要使用的 GPU ID，如 0 或 1")
parser.add_argument("--ckpt_name", type=str, default="mobilenet_hybrid_best.pth",
                    help="权重文件名（默认 mobilenet_hybrid_best.pth），位于 save_dir/exp_name/ 下")
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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
import pandas as pd
import numpy as np

# ==============================================================================
# PART 0: 实验集配置 (EXPERIMENTS) 
# ==============================================================================
EXPERIMENTS = {
    "exp1": {
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/RSCD/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/ACDC/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/KITTI-360/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/IDD/processed_data/train/local_image_final"
        ],
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/DeepScene/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/TAS500/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/RELLIS/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/RELLIS/processed_data/valid/local_image_final"
        ]
    },
    "exp2": {
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/RSCD/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/ACDC/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/KITTI-360/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/IDD/processed_data/train/local_image_final"
        ],
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/test",
            "/8TBHDD3/tht/TerraData/RUGD/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/RUGD/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/GOOSE/goose/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/GOOSE/gooseEx/processed_data/valid/local_image_final"
        ]
    },
    "exp3": {
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/DeepScene/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/WildScenes/processed_data/local_image_final",
            "/8TBHDD3/tht/TerraData/TAS500/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/RELLIS/processed_data/train/local_image_final"
        ],
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/RSCD/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/KITTI-360/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/IDD/processed_data/valid/local_image_final"
        ]
    },
    "exp4": {
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/DeepScene/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/WildScenes/processed_data/local_image_final",
            "/8TBHDD3/tht/TerraData/TAS500/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/RELLIS/processed_data/train/local_image_final"
        ],
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/test",
            "/8TBHDD3/tht/TerraData/RUGD/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/RUGD/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/GOOSE/goose/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/GOOSE/gooseEx/processed_data/valid/local_image_final"
        ]
    },
    "exp5": {
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/test",
            "/8TBHDD3/tht/TerraData/RUGD/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/GOOSE/goose/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/GOOSE/gooseEx/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/ORFD/processed_data/train/local_image_final"
        ],
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/RSCD/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/KITTI-360/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/IDD/processed_data/valid/local_image_final"
        ]
    },
    "exp6": {
        "train_dirs": [
            "/8TBHDD3/tht/TerraData/TerraPOSS/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/Jackal/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/VAST/processed_data/local_image_final_split/test",
            "/8TBHDD3/tht/TerraData/RUGD/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/GOOSE/goose/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/GOOSE/gooseEx/processed_data/train/local_image_final",
            "/8TBHDD3/tht/TerraData/ORFD/processed_data/train/local_image_final"
        ],
        "test_dirs": [
            "/8TBHDD3/tht/TerraData/DeepScene/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/TAS500/processed_data/valid/local_image_final",
            "/8TBHDD3/tht/TerraData/RELLIS/processed_data/test/local_image_final",
            "/8TBHDD3/tht/TerraData/RELLIS/processed_data/valid/local_image_final"
        ]
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
ALIAS_MAP = {"grass floor": "grass", "coated_floor": "coated floor"}

CONFIG = {
    "json_name": "local_label.json",
    "batch_size": 512,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 8,
    "img_size": 224,
    "save_dir": "./mobilenet_checkpoints/benchmark2"
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

        for folder_path in dir_list:
            if not os.path.exists(folder_path):
                continue
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
            items = data.get("items", [])
            for item in items:
                img_name = item.get("name")
                label_str = item.get("material")
                if img_name and label_str:
                    label_str = label_str.lower()
                    if label_str in ALIAS_MAP:
                        label_str = ALIAS_MAP[label_str]
                    if label_str in GLOBAL_LABEL_MAP:
                        full_img_path = os.path.join(folder_path, img_name)
                        if os.path.exists(full_img_path):
                            self.samples.append((full_img_path, label_str))
                            self.class_counts[label_str] += 1
        except Exception as e:
            pass

    def _load_folder_mode(self, root_path):
        for class_name in os.listdir(root_path):
            class_dir = os.path.join(root_path, class_name)
            if not os.path.isdir(class_dir):
                continue
            label_str = class_name.lower()
            if label_str in ALIAS_MAP:
                label_str = ALIAS_MAP[label_str]
            if label_str in GLOBAL_LABEL_MAP:
                files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                for img_file in files:
                    full_img_path = os.path.join(class_dir, img_file)
                    self.samples.append((full_img_path, label_str))
                    self.class_counts[label_str] += 1

    def __len__(self):
        return len(self.samples)

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

def resolve_checkpoint(exp_dir: str, ckpt_name: str) -> str:
    preferred = os.path.join(exp_dir, ckpt_name)
    if os.path.exists(preferred):
        return preferred
    pths = [os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if f.endswith(".pth")]
    if len(pths) == 0:
        raise FileNotFoundError(f"❌ 在 {exp_dir} 下未找到任何 .pth 权重文件！")
    pths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return pths[0]

# ==========================================
# PART 4: 核心执行函数 (执行单个实验)
# ==========================================
def run_single_experiment(exp_name, args):
    exp_cfg = EXPERIMENTS[exp_name]
    CONFIG["test_dirs"] = exp_cfg["test_dirs"]

    exp_save_dir = os.path.join(CONFIG["save_dir"], exp_name)
    if not os.path.isdir(exp_save_dir):
        print(f"⚠️ 跳过 {exp_name}: 找不到实验目录 {exp_save_dir}")
        return

    try:
        ckpt_path = resolve_checkpoint(exp_save_dir, args.ckpt_name)
    except Exception as e:
        print(f"⚠️ 跳过 {exp_name}: {e}")
        return

    print(f"\n=======================================================")
    print(f"🧪 开始测试: {exp_name} | 权重: {os.path.basename(ckpt_path)}")
    print(f"=======================================================\n")

    test_transforms = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = TerraHybridDataset(CONFIG["test_dirs"], CONFIG["json_name"],
                                      transform=test_transforms, is_train=False)
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG["batch_size"],
        shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True
    )

    global_num_classes = len(ALL_TERRAIN_CLASSES)
    model = VisionNet(num_classes=global_num_classes, head_type="wide").to(CONFIG["device"])

    state = torch.load(ckpt_path, map_location=CONFIG["device"])
    model.load_state_dict(state)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Testing {exp_name}"):
            inputs = inputs.to(CONFIG["device"])
            labels = labels.to(CONFIG["device"])
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    unique_classes = sorted(list(set(all_labels)))
    target_names = [ALL_TERRAIN_CLASSES[i] for i in unique_classes]

    # 计算指标
    final_acc = accuracy_score(all_labels, all_preds)
    report_dict = classification_report(
        all_labels, all_preds, labels=unique_classes,
        target_names=target_names, digits=4, zero_division=0, output_dict=True
    )
    
    macro_f1 = report_dict["macro avg"]["f1-score"]
    macro_recall = report_dict["macro avg"]["recall"]

    print(f"\n✅ [{exp_name}] Final Acc: {final_acc:.4f} | Macro F1: {macro_f1:.4f} | Macro Recall: {macro_recall:.4f}")

    # ==========================================
    # 绘制并保存混淆矩阵 (Confusion Matrix)
    # 自动保存 full + ppt 两个版本
    # 重点：尽可能放大横纵坐标标签和坐标轴标题，
    #      单元格数字字号相对更小
    # ==========================================
    cm = confusion_matrix(all_labels, all_preds, labels=unique_classes)

    def save_confusion_matrix_versions(cm, target_names, exp_name, exp_save_dir, cmap="Blues"):
        n_cls = len(target_names)

        def _draw_and_save(version="full"):
            # -----------------------------
            # 1) 根据版本动态设置参数
            # -----------------------------
            if version == "full":
                show_numbers = True
                dpi = 700

                fig_w = max(24, n_cls * 1.15)
                fig_h = max(20, n_cls * 1.05)

                title_fs = max(26, min(38, int(fig_w * 1.20)))
                label_fs = max(26, min(34, int(fig_w * 1.00)))

                # 横纵类别标签进一步放大
                tick_fs = max(36, min(36, int(520 / max(n_cls, 1))))

                # 单元格数字继续保持偏小
                annot_fs = max(16, min(28, int(420 / max(n_cls, 1))))

                cbar_fs = max(14, min(20, tick_fs - 6))

                xtick_rot = 45
                ytick_rot = 0

            elif version == "ppt":
                show_numbers = True
                dpi = 600

                # 画布再放大
                fig_w = max(22, n_cls * 1.20)
                fig_h = max(18, n_cls * 1.05)

                title_fs = max(24, min(32, int(fig_w * 1.10)))
                label_fs = max(22, min(30, int(fig_w * 0.95)))

                # 类别名字号很大
                tick_fs = max(36, min(48, int(420 / max(n_cls, 1))))

                # 数字也较大
                annot_fs = max(24, min(32, int(320 / max(n_cls, 1))))
                cbar_fs = max(14, min(20, tick_fs - 8))

                xtick_rot = 60
                ytick_rot = 0

            else:
                raise ValueError(f"Unknown version: {version}")

            # -----------------------------
            # 2) 开始绘图
            # -----------------------------
            plt.figure(figsize=(fig_w, fig_h))

            ax = sns.heatmap(
                cm,
                annot=show_numbers,
                fmt='d',
                cmap=cmap,
                cbar=True,
                xticklabels=target_names,
                yticklabels=target_names,
                annot_kws={"size": annot_fs, "weight": "normal"} if show_numbers else None,
                square=True,
                linewidths=0.35,
                linecolor='lightgray'
            )

            # 标题和轴标题尽量大
            ax.set_title(
                f"Confusion Matrix: {exp_name} (Zero-Shot Transfer)",
                fontsize=title_fs,
                fontweight='bold',
                pad=20
            )
            ax.set_ylabel(
                "True Target (Ground Truth)",
                fontsize=label_fs,
                fontweight='bold',
                labelpad=16
            )
            ax.set_xlabel(
                "Predicted Label",
                fontsize=label_fs,
                fontweight='bold',
                labelpad=16
            )

            # 横纵坐标标签尽可能放大
            ax.tick_params(axis='x', labelsize=tick_fs, rotation=xtick_rot, pad=6)
            ax.tick_params(axis='y', labelsize=tick_fs, rotation=ytick_rot, pad=6)

            # 对长标签做换行，避免互相压住
            wrapped_target_names = [name.replace(" ", "\n") for name in target_names]

            ax.set_xticklabels(wrapped_target_names)
            ax.set_yticklabels(wrapped_target_names)

            # 压缩同一标签内部的换行行距
            for lbl in ax.get_xticklabels():
                lbl.set_linespacing(0.8)   # 默认大约是 1.2，0.6~0.8 常用
            for lbl in ax.get_yticklabels():
                lbl.set_linespacing(0.8)

            # 精细控制旋转和对齐
            plt.setp(ax.get_xticklabels(), rotation=xtick_rot, ha='right', rotation_mode='anchor')
            plt.setp(ax.get_yticklabels(), rotation=ytick_rot, va='center')

            # colorbar 字号
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=cbar_fs)

            plt.tight_layout()

            save_path = os.path.join(
                exp_save_dir,
                f"{exp_name}_confusion_matrix_{version}.png"
            )
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            plt.close()

            print(f"🖼️ [混淆矩阵保存 - {version}] {save_path}")

        # 自动保存两个版本
        # _draw_and_save("full")
        _draw_and_save("ppt")

    # 调用：生成 full + ppt 两个版本
    save_confusion_matrix_versions(
        cm=cm,
        target_names=target_names,
        exp_name=exp_name,
        exp_save_dir=exp_save_dir,
        cmap="Blues"
    )

    # ==========================================
    # 更新汇总表 (新增 Macro 指标多行记录 - 修复对齐问题)
    # ==========================================
    summary_csv_path = os.path.join(CONFIG["save_dir"], "all_experiments_summary.csv")
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    exp_columns = list(EXPERIMENTS.keys())

    if os.path.exists(summary_csv_path):
        try:
            # 尝试读取，如果有问题则重新创建
            df_summary = pd.read_csv(summary_csv_path, index_col=0)
            # 检查列名是否正常，如果因为旧格式错位了，则重置表
            if not all(col in df_summary.columns for col in exp_columns):
                print("⚠️ 发现旧版或格式错位的 CSV，将自动重建汇总表...")
                df_summary = pd.DataFrame(columns=exp_columns)
        except Exception:
            df_summary = pd.DataFrame(columns=exp_columns)
    else:
        df_summary = pd.DataFrame(columns=exp_columns)

    # 确保列顺序正确且只包含定义的 exp
    for col in exp_columns:
        if col not in df_summary.columns:
            df_summary[col] = float("nan")
    df_summary = df_summary[exp_columns]

    # 确保所需的行(指标)存在
    for metric in ["Accuracy", "Macro_F1", "Macro_Recall"]:
        if metric not in df_summary.index:
            df_summary.loc[metric] = float("nan")

    # 填入最新数据
    df_summary.at["Accuracy", exp_name] = final_acc
    df_summary.at["Macro_F1", exp_name] = macro_f1
    df_summary.at["Macro_Recall", exp_name] = macro_recall

    # 💡 核心修复：添加 index_label="Metric"，锁定表头对齐
    df_summary.to_csv(summary_csv_path, index=True, index_label="Metric", float_format="%.6f")
    print(f"📈 [汇总更新] 指标已录入并对齐: {summary_csv_path}")

# ==========================================
# PART 5: 主程序入口 
# ==========================================
def main(args):
    if args.exp_name.lower() == "all":
        print("🚀 触发批量测试模式，将依次运行 exp1 到 exp6...")
        exps_to_run = list(EXPERIMENTS.keys())
    else:
        if args.exp_name not in EXPERIMENTS:
            raise ValueError(f"❌ 实验名称 '{args.exp_name}' 未在 EXPERIMENTS 中定义！")
        exps_to_run = [args.exp_name]

    for exp in exps_to_run:
        run_single_experiment(exp, args)
        
    print("\n🎉 所有指定测试流程执行完毕！")

if __name__ == "__main__":
    main(args)