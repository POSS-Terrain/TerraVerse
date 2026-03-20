import os
import argparse
import json
import torch
import numpy as np  
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix # 💡 新增 confusion_matrix
import matplotlib.pyplot as plt  # 💡 新增画图包
import seaborn as sns            # 💡 新增画图包
from tqdm import tqdm
from collections import Counter
import clip
import torch.nn.functional as F
from peft import PeftModel

# ==============================================================================
# 1. 参数解析与 GPU 设置
# ==============================================================================
parser = argparse.ArgumentParser(description="TerraCLIP 纯测试脚本 (支持单实验或批量遍历)")
parser.add_argument("--benchmark", type=str, required=True, help="Benchmark 文件夹名称 (例如 benchmark1, benchmark2)")
parser.add_argument("--exp_name", type=str, required=True, help="实验名称 (例如 exp1, exp2)，输入 'all' 则自动遍历")
parser.add_argument("--gpu", type=str, default="0", help="指定要使用的 GPU ID，如 0 或 1")
parser.add_argument("--ckpt_root", type=str, default="./clip_checkpoints", help="Checkpoints 的根目录")
args, _ = parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# ==============================================================================
# 2. 全局配置与标签定义
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
            "/8TBHDD3/tht/TerraData/RELLIS/processed_data/test/local_image_final","/8TBHDD3/tht/TerraData/RELLIS/processed_data/valid/local_image_final"
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
            "/8TBHDD3/tht/TerraData/RUGD/processed_data/test/local_image_final","/8TBHDD3/tht/TerraData/RUGD/processed_data/valid/local_image_final",
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
            "/8TBHDD3/tht/TerraData/RUGD/processed_data/test/local_image_final","/8TBHDD3/tht/TerraData/RUGD/processed_data/valid/local_image_final",
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
            "/8TBHDD3/tht/TerraData/RELLIS/processed_data/test/local_image_final","/8TBHDD3/tht/TerraData/RELLIS/processed_data/valid/local_image_final"
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
# 3. 数据集类
# ==============================================================================
class TerraHybridCLIPDataset(Dataset):
    def __init__(self, dir_list, transform=None, max_length=77):
        self.transform = transform
        self.max_length = max_length
        self.samples = [] 
        self.class_counts = Counter() 
        self.target_json = "local_label.json"
        
        for folder_path in dir_list:
            if not os.path.exists(folder_path):
                continue
            
            json_path = os.path.join(folder_path, self.target_json)
            if os.path.exists(json_path):
                self._load_test_labels(folder_path, json_path)

    def _load_test_labels(self, folder_path, json_path):
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
                        if not os.path.exists(full_img_path): 
                            continue
                        self.samples.append({
                            "image": full_img_path,
                            "label_idx": GLOBAL_LABEL_MAP[label_str],
                            "coarse_text": "",
                            "fine_text": ""
                        })
                        self.class_counts[label_str] += 1
        except Exception as e:
            pass

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
# 4. 推理核心逻辑
# ==============================================================================
def evaluate(model, loader, device, all_labels):
    model.eval()
    
    # 1. 粗粒度文本
    coarse_prompts = [f"The terrain material is {label}." for label in all_labels]
    coarse_tokens = clip.tokenize(coarse_prompts, truncate=True).to(device)
    
    # 2. 细粒度属性
    fine_attributes = {
        "weather": ["sunny", "cloudy", "rainy", "foggy", "snowy", "unknown"],
        "lighting": ["strong sunlight", "low light", "shadowed", "dark"],
        "moisture": ["dry", "moist", "wet"],
        "smoothness": ["smooth", "slightly uneven", "severely uneven"],
        "friction_hint": ["high", "medium", "low"],
        "traversability_hint": ["traversable", "non-traversable"]
    }
    
    K = len(all_labels)
    fine_prompts_flat = []
    
    for label in all_labels:
        for attr_key, attr_vals in fine_attributes.items():
            for val in attr_vals:
                fine_prompts_flat.append(f"The terrain material is {label}, characterized by {attr_key} being {val}.")
                
    M = len(fine_prompts_flat) // K  
    
    with torch.no_grad():
        coarse_features = model.encode_text(coarse_tokens)
        coarse_features = F.normalize(coarse_features, dim=-1)
        
        fine_tokens = clip.tokenize(fine_prompts_flat, truncate=True).to(device)
        fine_features_flat = model.encode_text(fine_tokens)
        fine_features_flat = F.normalize(fine_features_flat, dim=-1)
        
    all_preds = []
    all_targets = []
    lambda_weight = 0.3 
    
    log_K = np.log(K)
    
    with torch.no_grad():
        logit_scale = model.logit_scale.exp().item()
        for images, _, _, labels in tqdm(loader, desc="Validating (Dynamic Weighting)"):
            images = images.to(device)
            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            
            logits_coarse = logit_scale * (image_features @ coarse_features.t()) 
            probs_coarse = F.softmax(logits_coarse, dim=-1) 
            
            logits_fine = logit_scale * (image_features @ fine_features_flat.t()) 
            probs_fine_flat = F.softmax(logits_fine, dim=-1) 
            probs_fine = probs_fine_flat.view(-1, K, M)
            
            sum_score = probs_fine.sum(dim=-1) 
            max_score = probs_fine.max(dim=-1).values 
            fine_score = sum_score + lambda_weight * max_score 
            
            entropy = -torch.sum(probs_coarse * torch.log(probs_coarse + 1e-9), dim=-1) 
            alpha = entropy / log_K 
            alpha = alpha.unsqueeze(-1) 
            
            final_scores = (1.0 - alpha) * probs_coarse + alpha * fine_score 
            preds = final_scores.argmax(dim=-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    return acc, all_targets, all_preds

# ==============================================================================
# 5. 核心单实验执行函数 (支持批量遍历提取)
# ==============================================================================
def run_single_experiment(exp_name, base_model, args):
    exp_cfg = EXPERIMENTS[exp_name]
    CONFIG["test_dirs"] = exp_cfg["test_dirs"]
    
    ckpt_path = os.path.join(args.ckpt_root, args.benchmark, exp_name, "clip_hybrid_best")
    if not os.path.exists(ckpt_path):
        print(f"⚠️ 跳过 {exp_name}: 未找到 Checkpoint: {ckpt_path}")
        return

    output_dir = os.path.join(args.ckpt_root, args.benchmark, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n=======================================================")
    print(f"🚀 启动评估: {args.benchmark} | {exp_name}")
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

    print(f"加载 LoRA 适配器来自 {ckpt_path}...")
    model = PeftModel.from_pretrained(base_model, ckpt_path)
    model.to(device)

    final_acc, targets, preds = evaluate(model, test_loader, device, ALL_TERRAIN_CLASSES)
    
    unique_classes = sorted(list(set(targets)))
    target_names = [ALL_TERRAIN_CLASSES[i] for i in unique_classes]
    
    # 获取详细 report 用于提取 Macro 指标
    report_dict = classification_report(targets, preds, labels=unique_classes, target_names=target_names, digits=4, zero_division=0, output_dict=True)
    macro_f1 = report_dict["macro avg"]["f1-score"]
    macro_recall = report_dict["macro avg"]["recall"]
    
    print(f"\n✅ [{exp_name}] Final Acc: {final_acc:.4f} | Macro F1: {macro_f1:.4f} | Macro Recall: {macro_recall:.4f}")

    # 保存常规分类报告 CSV
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.loc['Final_Test_Accuracy'] = [final_acc, None, None, None]
    csv_path = os.path.join(output_dir, f"revise_{exp_name}_test_results.csv")
    df_report.to_csv(csv_path, index=True)

        # ==========================================
    # 🚩 绘制并保存混淆矩阵热力图
    # 自动保存 full + ppt 两个版本
    # ==========================================
    cm = confusion_matrix(targets, preds, labels=unique_classes)

    def save_confusion_matrix_versions(cm, target_names, exp_name, output_dir, cmap="Oranges"):
        n_cls = len(target_names)

        def _draw_and_save(version="full"):
            # -----------------------------
            # 1) 根据版本动态设置参数
            # -----------------------------
            if version == "full":
                # full版：单独查看/论文附录，保留格子数字
                show_numbers = True
                dpi = 600

                fig_w = max(18, n_cls * 0.9)
                fig_h = max(14, n_cls * 0.8)

                title_fs = max(20, min(30, int(fig_w * 1.2)))
                label_fs = max(16, min(24, int(fig_w * 0.9)))
                tick_fs  = max(12, min(20, int(220 / max(n_cls, 1))))
                annot_fs = max(10, min(18, int(200 / max(n_cls, 1))))
                cbar_fs  = max(12, tick_fs)

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
                annot_kws={"size": annot_fs, "weight": "bold"} if show_numbers else None,
                square=True,
                linewidths=0.3,
                linecolor='lightgray'
            )

            ax.set_title(
                f"Confusion Matrix: {exp_name} (TerraCLIP Zero-Shot)",
                fontsize=title_fs,
                fontweight='bold',
                pad=16
            )
            ax.set_ylabel(
                "True Target (Ground Truth)",
                fontsize=label_fs,
                fontweight='bold',
                labelpad=10
            )
            ax.set_xlabel(
                "Predicted Label",
                fontsize=label_fs,
                fontweight='bold',
                labelpad=10
            )

            ax.tick_params(axis='x', labelsize=tick_fs, rotation=xtick_rot)
            ax.tick_params(axis='y', labelsize=tick_fs, rotation=ytick_rot)

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


            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=cbar_fs)

            plt.tight_layout()

            save_path = os.path.join(output_dir, f"{exp_name}_confusion_matrix_{version}.png")
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            plt.close()

            print(f"🖼️ [混淆矩阵保存 - {version}] {save_path}")

        # 自动保存两个版本
        # _draw_and_save("full")
        _draw_and_save("ppt")

    # 调用：保存 full + ppt 两个版本
    save_confusion_matrix_versions(
        cm=cm,
        target_names=target_names,
        exp_name=exp_name,
        output_dir=output_dir,
        cmap="Oranges"
    )

    # ==============================================================================
    # 🚩 更新汇总表：包含 Accuracy, Macro F1, Macro Recall
    # ==============================================================================
    summary_dir = os.path.join(args.ckpt_root, args.benchmark)
    summary_csv_path = os.path.join(summary_dir, "all_experiments_summary.csv")
    os.makedirs(summary_dir, exist_ok=True)
    exp_columns = list(EXPERIMENTS.keys())
    
    if os.path.exists(summary_csv_path):
        df_summary = pd.read_csv(summary_csv_path, index_col=0)
        for col in exp_columns:
            if col not in df_summary.columns:
                df_summary[col] = float('nan')
    else:
        df_summary = pd.DataFrame(columns=exp_columns)

    for metric in ["Accuracy", "Macro_F1", "Macro_Recall"]:
        if metric not in df_summary.index:
            df_summary.loc[metric] = float("nan")

    df_summary.at["Accuracy", exp_name] = final_acc
    df_summary.at["Macro_F1", exp_name] = macro_f1
    df_summary.at["Macro_Recall", exp_name] = macro_recall

    df_summary.to_csv(summary_csv_path, index=True, float_format='%.6f')
    print(f"📈 [汇总更新] 指标已录入总表: {summary_csv_path}")

# ==============================================================================
# 6. 主程序入口
# ==============================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n[Init] 预加载基础 CLIP 模型 (ViT-B/32)...")
    base_model, _ = clip.load(CONFIG["model_name"], device=device)
    base_model = base_model.float() 

    if args.exp_name.lower() == "all":
        print("🚀 触发批量测试模式，将依次运行 exp1 到 exp6...")
        exps_to_run = list(EXPERIMENTS.keys())
    else:
        if args.exp_name not in EXPERIMENTS:
            raise ValueError(f"❌ 实验名称 '{args.exp_name}' 未在 EXPERIMENTS 中定义！")
        exps_to_run = [args.exp_name]

    for exp in exps_to_run:
        run_single_experiment(exp, base_model, args)
        
    print("\n🎉 所有指定测试流程执行完毕！")

if __name__ == "__main__":
    main()