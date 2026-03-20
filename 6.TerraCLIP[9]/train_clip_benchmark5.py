import os
import argparse

# ==============================================================================
# 🚨 极其重要：必须在导入 torch 前完成参数解析和 GPU 环境变量设置！
# ==============================================================================
parser = argparse.ArgumentParser(description="多卡并发训练 TerraCLIP - Few-Shot Adaptation (MLP)")
parser.add_argument("--exp_name", type=str, required=True, help="实验名称，如 exp1, exp2")
parser.add_argument("--gpu", type=str, default="0", help="指定要使用的 GPU ID，如 0 或 1")
args, _ = parser.parse_known_args()

# 动态设置系统可见的 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# ==============================================================================
# --- 环境设置完毕后，再导入深度学习包 ---
# ==============================================================================
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from collections import Counter
import clip
import wandb
from peft import LoraConfig, get_peft_model, PeftModel
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import numpy as np

# ==============================================================================
# PART 0: 实验集配置 (EXPERIMENTS) 
# ==============================================================================
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

# ==============================================================================
# PART 1: 全局配置与标签定义
# ==============================================================================
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
    "lr": 5e-5,                 
    "base_epochs": 50,
    "fewshot_epochs": 50,
    "num_workers": 8,
    "img_size": 224,            
    "coarse_weight": 0.6,       
    "model_name": "ViT-B/32",
    "fine_tune_method": "lora", 
    "save_dir": "./terraclip_checkpoints/benchmark5",
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "wandb_project": "TerraVerse-TerraCLIP-Benchmark5",
}

# ==============================================================================
# PART 2: 数据集类与 1% Few-Shot 采样器
# ==============================================================================
class TerraHybridCLIPDataset(Dataset):
    def __init__(self, dir_list, transform=None, dataset_role="base_train"):
        self.transform = transform
        self.samples = [] 
        self.class_counts = Counter() 
        self.dataset_role = dataset_role
        
        self.target_json = "annotations.json" if dataset_role == "base_train" else "local_label.json"
        
        for folder_path in dir_list:
            if not os.path.exists(folder_path): continue
            
            json_path = os.path.join(folder_path, self.target_json)
            if os.path.exists(json_path):
                if dataset_role == "base_train":
                    self._load_train_annotations(folder_path, json_path)
                else:
                    self._load_test_labels(folder_path, json_path)

    def _load_train_annotations(self, folder_path, json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                img_name = item.get("name")
                final_ann = item.get("final_annotation", {})
                material = final_ann.get("material", "").lower()
                
                if not img_name or not material: continue
                if material in ALIAS_MAP: material = ALIAS_MAP[material]
                
                if material in GLOBAL_LABEL_MAP:
                    full_img_path = os.path.join(folder_path, img_name)
                    if not os.path.exists(full_img_path): continue
                        
                    global_context = final_ann.get("global_context", {})
                    vis_attrs = final_ann.get("visual_attributes", {})
                    vis_hints = final_ann.get("visual_physical_hints", {})
                    
                    weather = global_context.get("weather", "unknown")
                    lighting = global_context.get("lighting", "unknown")
                    smoothness = vis_attrs.get("smoothness", "unknown")
                    moisture = vis_attrs.get("moisture", "unknown")
                    trav_hint = vis_hints.get("traversability_hint", "unknown")
                    fric_hint = vis_hints.get("friction_hint", "unknown")

                    coarse_text = f"The terrain material is {material}."
                    fine_text = (f"The terrain material is {material}, "
                                 f"traversability_hint is {trav_hint}, "
                                 f"weather is {weather}, lighting is {lighting}, "
                                 f"smoothness is {smoothness}, moisture is {moisture}, "
                                 f"friction_hint is {fric_hint}.")
                    
                    self.samples.append({
                        "image": full_img_path,
                        "label_idx": GLOBAL_LABEL_MAP[material],
                        "coarse_text": coarse_text,
                        "fine_text": fine_text,
                        "label_name": material
                    })
                    self.class_counts[material] += 1
        except Exception as e: pass

    def _load_test_labels(self, folder_path, json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            items = data.get("items", [])
            for item in items:
                img_name = item.get("name")
                label_str = item.get("label", "").lower()
                
                if not img_name or not label_str: continue
                if label_str in ALIAS_MAP: label_str = ALIAS_MAP[label_str]
                
                if label_str in GLOBAL_LABEL_MAP:
                    full_img_path = os.path.join(folder_path, img_name)
                    if not os.path.exists(full_img_path): continue
                        
                    coarse_text = f"The terrain material is {label_str}."
                    self.samples.append({
                        "image": full_img_path,
                        "label_idx": GLOBAL_LABEL_MAP[label_str],
                        "coarse_text": coarse_text,
                        "fine_text": coarse_text, 
                        "label_name": label_str
                    })
                    self.class_counts[label_str] += 1
        except Exception as e: pass

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        try:
            image = Image.open(item['image']).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224))
        if self.transform:
            image = self.transform(image)
            
        # 🟡 Bug修复: 移除不正确的字符截断, clip.tokenize(truncate=True) 会完美处理
        coarse_text = item['coarse_text']
        fine_text = item['fine_text']
        return image, coarse_text, fine_text, item['label_idx']

class DatasetWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        image, coarse_text, fine_text, label_idx = self.subset[index]
        if self.transform:
            image = self.transform(image)
        return image, coarse_text, fine_text, label_idx

    def __len__(self):
        return len(self.subset)

def get_stratified_few_shot_subset(dataset, fraction=0.01):
    labels = [dataset.samples[i]["label_name"] for i in range(len(dataset))]
    unique_labels = list(set(labels))
    selected_indices = []
    
    np.random.seed(42) 
    for lbl in unique_labels:
        lbl_indices = [i for i, x in enumerate(labels) if x == lbl]
        n_select = max(1, int(len(lbl_indices) * fraction)) 
        selected = np.random.choice(lbl_indices, n_select, replace=False)
        selected_indices.extend(selected)
        
    print(f"   🔍 Few-Shot 1% 采样完成: 共抽取了 {len(selected_indices)} 张图片。")
    return Subset(dataset, selected_indices)


# ==============================================================================
# PART 3: 核心网络结构与训练验证逻辑
# ==============================================================================

# 🌟 严格对应论文 Benchmark 3 的三层 MLP 分类头
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        # 典型的三层 MLP 结构：Linear -> ReLU -> Linear -> ReLU -> Linear
        hidden_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# --- 阶段一：源域上的对比学习训练 ---
def train_base_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training (Base)") 
    
    for images, coarse_texts, fine_texts, labels in pbar:
        images = images.to(device)
        # clip.tokenize 原生自带 truncate=True，自动处理 Token 截断
        coarse_tokens = clip.tokenize(coarse_texts, truncate=True).to(device)
        fine_tokens = clip.tokenize(fine_texts, truncate=True).to(device)
        
        optimizer.zero_grad()
        # 🟡 Bug修复: 统一使用 autocast()，增强向下兼容性
        with autocast():
            image_features = model.encode_image(images)
            coarse_features = model.encode_text(coarse_tokens)
            fine_features = model.encode_text(fine_tokens)
            
            image_features = F.normalize(image_features, dim=-1)
            coarse_features = F.normalize(coarse_features, dim=-1)
            fine_features = F.normalize(fine_features, dim=-1)
            
            logit_scale = model.logit_scale.exp()
            
            logits_img_coarse = logit_scale * image_features @ coarse_features.t()
            logits_coarse_img = logits_img_coarse.t()
            labels_idx = torch.arange(len(images), device=device)
            loss_coarse = (F.cross_entropy(logits_img_coarse, labels_idx) + 
                           F.cross_entropy(logits_coarse_img, labels_idx)) / 2

            logits_img_fine = logit_scale * image_features @ fine_features.t()
            logits_fine_img = logits_img_fine.t()
            loss_fine = (F.cross_entropy(logits_img_fine, labels_idx) + 
                         F.cross_entropy(logits_fine_img, labels_idx)) / 2
            
            loss = CONFIG["coarse_weight"] * loss_coarse + (1 - CONFIG["coarse_weight"]) * loss_fine

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}") 
        
    return total_loss / len(loader)


# --- 阶段一：Zero-Shot 验证 ---
def evaluate_base(model, loader, device, all_labels):
    model.eval()
    
    # 按照多粒度计算 Prototype（简化的粗粒度 Zero-shot，用于阶段一监控）
    text_prompts = [f"The terrain material is {label}." for label in all_labels]
    text_tokens = clip.tokenize(text_prompts).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, _, _, labels in tqdm(loader, desc="Zero-Shot Eval"):
            images = images.to(device)
            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            
            similarity = (100.0 * image_features @ text_features.t())
            preds = similarity.argmax(dim=-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    return acc, all_targets, all_preds


# 🌟 阶段二 & 三 的 MLP 盲测验证
def evaluate_mlp(base_model, mlp_head, loader, device):
    base_model.eval()
    mlp_head.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, _, _, labels in tqdm(loader, desc="MLP Eval"):
            images = images.to(device)
            
            # 1. 提取冻结后的图像特征
            with autocast():
                image_features = base_model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
            
            # 2. 通过 MLP 分类头
            logits = mlp_head(image_features.float())
            preds = logits.argmax(dim=-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    return acc, all_targets, all_preds


# ==============================================================================
# PART 4: 主函数
# ==============================================================================
def main(args):
    if args.exp_name not in EXPERIMENTS:
        raise ValueError(f"❌ 实验名称 '{args.exp_name}' 未在 EXPERIMENTS 中定义！")
    
    exp_cfg = EXPERIMENTS[args.exp_name]
    
    # 🔴 修复致命 KeyError Bug：映射三个正确的键名到 CONFIG
    CONFIG["source_train_dirs"] = exp_cfg["source_train_dirs"]
    CONFIG["target_train_dirs"] = exp_cfg["target_train_dirs"]
    CONFIG["target_test_dirs"] = exp_cfg["target_test_dirs"]
    
    exp_save_dir = os.path.join(CONFIG["save_dir"], args.exp_name)
    os.makedirs(exp_save_dir, exist_ok=True)
    best_base_dir = os.path.join(exp_save_dir, "clip_base_best")
    best_mlp_path = os.path.join(exp_save_dir, "mlp_fewshot_best.pth")
    
    wandb.init(project=CONFIG["wandb_project"], config=CONFIG, name=f"run_{args.exp_name}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=======================================================")
    print(f"🚀 启动 TerraCLIP (MLP Few-Shot) 实验: {args.exp_name} | GPU: {args.gpu}")
    print(f"=======================================================\n")
    
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)

    train_transform_safe = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])), 
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(clip_mean, clip_std)
    ])
    
    valid_transform_safe = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(clip_mean, clip_std)
    ])

    # ------------------ 数据集准备 ------------------
    # 1. Base 训练集 (Source) -> 提取细粒度特征
    full_source_dataset = TerraHybridCLIPDataset(CONFIG["source_train_dirs"], transform=None, dataset_role="base_train")
    total_size = len(full_source_dataset)
    train_size = int(0.8 * total_size)
    valid_size = total_size - train_size
    raw_train_set, raw_valid_set = random_split(full_source_dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(42))
    
    source_train_dataset = DatasetWrapper(raw_train_set, transform=train_transform_safe)
    source_valid_dataset = DatasetWrapper(raw_valid_set, transform=valid_transform_safe)

    # 2. Few-Shot 目标域训练集 (1% Target) -> 用于微调 MLP，不需要文本
    full_target_train_dataset = TerraHybridCLIPDataset(CONFIG["target_train_dirs"], transform=None, dataset_role="fewshot_train")
    few_shot_raw_subset = get_stratified_few_shot_subset(full_target_train_dataset, fraction=0.01)
    target_fewshot_dataset = DatasetWrapper(few_shot_raw_subset, transform=train_transform_safe)

    # 3. 目标域测试集 (Target Test) -> 用于最终评估
    target_test_dataset = TerraHybridCLIPDataset(CONFIG["target_test_dirs"], transform=valid_transform_safe, dataset_role="test")

    source_train_loader = DataLoader(source_train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)
    source_valid_loader = DataLoader(source_valid_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
    
    fs_batch_size = min(32, len(target_fewshot_dataset))
    target_fewshot_loader = DataLoader(target_fewshot_dataset, batch_size=fs_batch_size, shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)
    
    target_test_loader  = DataLoader(target_test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
    
    # === 模型初始化 (Base) ===
    model, _ = clip.load(CONFIG["model_name"], device=device)
    model = model.float() # FP32 安全模式
    clip_embed_dim = model.visual.output_dim # 动态获取 CLIP 视觉特征维度 (ViT-B/32为512)
    
    print(f"Initializing LoRA for Base Pretraining (Rank={CONFIG['lora_rank']})...")
    config = LoraConfig(
        r=CONFIG["lora_rank"],
        lora_alpha=CONFIG["lora_alpha"],
        target_modules=["c_fc", "c_proj"],
        lora_dropout=CONFIG["lora_dropout"],
        bias="none"
    )
    model = get_peft_model(model, config)
    
    scaler = GradScaler() # 🟡 优化：采用更稳定的初始化方式

    # =======================================================
    # STAGE 1: Base Pretraining (源域上做图文对比学习)
    # =======================================================
    print(f"\n🚀 [阶段一] 开始 Base Pretraining (Epochs: {CONFIG['base_epochs']}) ...")
    optimizer_base = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=0.01)
    
    best_base_acc = 0.0
    patience = 5
    counter = 0 
    
    for epoch in range(CONFIG["base_epochs"]):
        train_loss = train_base_epoch(model, source_train_loader, optimizer_base, scaler, device)
        val_acc, _, _ = evaluate_base(model, source_valid_loader, device, ALL_TERRAIN_CLASSES)
        
        print(f"   Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
        wandb.log({"base_val_acc": val_acc, "epoch": epoch})
        
        if val_acc > best_base_acc + 0.001:
            best_base_acc = val_acc
            counter = 0 
            model.save_pretrained(best_base_dir)
        else:
            counter += 1
            if counter >= patience:
                print("   🛑 Base Pretraining Early stopping triggered!")
                break

    # =======================================================
    # STAGE 2: Few-Shot Fine-Tuning (冻结Backbone，训练 MLP 头)
    # =======================================================
    print(f"\n🚀 [阶段二] 开始 Few-Shot 1% MLP Fine-Tuning (Epochs: {CONFIG['fewshot_epochs']}) ...")
    
    # 1. 重新加载 Base Model 并完全冻结
    if os.path.exists(best_base_dir):
        base_model, _ = clip.load(CONFIG["model_name"], device=device)
        base_model = base_model.float() 
        # is_trainable=False 确保 LoRA 权重和 Backbone 彻底被冻结
        model = PeftModel.from_pretrained(base_model, best_base_dir, is_trainable=False)
        model.to(device)
    
    model.eval() # 切换为评估模式，关闭 Dropout 和 BatchNorm 等行为

    # 2. 初始化三层 MLP 分类头
    global_num_classes = len(ALL_TERRAIN_CLASSES)
    mlp_head = MLPClassifier(input_dim=clip_embed_dim, num_classes=global_num_classes).to(device)
    
    # 3. 仅对 MLP 使用交叉熵进行优化
    optimizer_mlp = optim.Adam(mlp_head.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(CONFIG["fewshot_epochs"]):
        mlp_head.train()
        epoch_loss = 0.0
        
        for images, _, _, labels in target_fewshot_loader: # 忽略返回的文本
            images, labels = images.to(device), labels.to(device)
            
            # 纯提取特征：不计算 Backbone 的梯度
            with torch.no_grad():
                with autocast():
                    image_features = model.encode_image(images)
                    image_features = F.normalize(image_features, dim=-1)
            
            optimizer_mlp.zero_grad()
            # 送入 MLP 进行分类预测
            logits = mlp_head(image_features.float())
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer_mlp.step()
            epoch_loss += loss.item()
            
        wandb.log({
            "fewshot_mlp_loss": epoch_loss / len(target_fewshot_loader), 
            "epoch": epoch + CONFIG["base_epochs"]
        })
            
    # 注意：极小样本下不切分验证集，直接跑满设定 Epoch 数并保存最终状态的权重
    torch.save(mlp_head.state_dict(), best_mlp_path)
    print(f">>> 💾 MLP Head Saved to {best_mlp_path}!")

    # =======================================================
    # STAGE 3: Final Testing (目标域全面盲测)
    # =======================================================
    print("\n=== 🏆 训练结束，加载最佳 MLP 模型进行最终盲测 ===")
    
    # 重新加载最优的 MLP 权重
    mlp_head.load_state_dict(torch.load(best_mlp_path, map_location=device))
    
    final_acc, targets, preds = evaluate_mlp(model, mlp_head, target_test_loader, device)
    print(f"Final Few-Shot MLP Test Accuracy: {final_acc:.4f}")
    
    unique_classes = sorted(list(set(targets)))
    target_names = [ALL_TERRAIN_CLASSES[i] for i in unique_classes]
    print(classification_report(targets, preds, labels=unique_classes, target_names=target_names, digits=4, zero_division=0))
    
    report_dict = classification_report(targets, preds, labels=unique_classes, target_names=target_names, digits=4, zero_division=0, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.loc['Final_Test_Accuracy'] = [final_acc, None, None, None]
    
    csv_filename = f"{args.exp_name}_mlp_fewshot_test_results.csv"
    csv_path = os.path.join(exp_save_dir, csv_filename)
    df_report.to_csv(csv_path, index=True)
    
    print(f"\n📊 [保存成功] 测试结果详细指标已导出至: {csv_path}")
    wandb.finish()

if __name__ == "__main__":
    main(args)