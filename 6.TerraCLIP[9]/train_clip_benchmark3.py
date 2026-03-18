import os
import argparse
import csv # 新增：导入 csv

# ==============================================================================
# 🚨 极其重要：必须在导入 torch 前完成参数解析和 GPU 环境变量设置！
# ==============================================================================
parser = argparse.ArgumentParser(description="多卡并发训练 TerraCLIP (Benchmark 3)")
parser.add_argument("--exp_name", type=str, required=True, help="实验名称，此处建议填 benchmark3")
parser.add_argument("--gpu", type=str, default="0", help="指定要使用的 GPU ID，如 0 或 1")
# 使用 parse_known_args 避免在这里报错，后续 main 中还可以继续用 args
args, _ = parser.parse_known_args()

# 动态设置系统可见的 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# ==============================================================================
# --- 环境设置完毕后，再导入沉重的深度学习包 ---
# ==============================================================================
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from collections import Counter
import clip
import wandb
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

import numpy as np
np.bool = np.bool_  # 打补丁：兼容 imgaug 的 numpy.bool 问题
import imgaug.augmenters as iaa

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

class ImgAugTransform:
    def __init__(self, prob=0.5):
        """
        prob: 触发增强的整体概率。0.5 表示一半的图片保持原样，一半会加上各种恶劣环境。
        """
        # 使用 iaa.Sometimes 控制整体概率
        # 使用 iaa.OneOf 确保每次只随机应用一种天气/退化效果，避免图片过度破坏
        self.aug = iaa.Sometimes(prob, iaa.OneOf([
            iaa.imgcorruptlike.MotionBlur(severity=(1, 2)),          # 运动模糊
            iaa.imgcorruptlike.Fog(severity=1),                      # 雾
            iaa.imgcorruptlike.Snow(severity=2),                     # 大雪
            iaa.Rain(drop_size=(0.10, 0.15), speed=(0.1, 0.2)),      # 雨
            iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.03)),# 雪点
            iaa.imgaug.augmenters.contrast.LinearContrast((0.5, 2.0), per_channel=0.5), # 对比度
            iaa.imgcorruptlike.Brightness(severity=(1, 2)),          # 亮度
            iaa.imgcorruptlike.Saturate(severity=(1, 3)),            # 饱和度
            iaa.Multiply((0.1, 0.5)), # 变暗 (这里将你原本的 slight/smooth/severe 合并为一个连续随机区间)
        ]))

    def __call__(self, img):
        # 1. PIL Image 转 NumPy 数组 (imgaug 需要的格式)
        img_np = np.array(img)
        
        # 2. 应用 imgaug 增强
        img_aug = self.aug.augment_image(img_np)
        
        # 3. NumPy 数组转回 PIL Image
        return Image.fromarray(img_aug)


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
    "epochs": 50,
    "num_workers": 8,
    "img_size": 224,            
    "coarse_weight": 0.6,       
    "model_name": "ViT-B/32",
    "fine_tune_method": "lora", 
    "save_dir": "./clip_checkpoints",
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "wandb_project": "CLIP-TerraVerse-Benchmark3", # 顺手帮你把项目名改为了 Benchmark3
}


# ==============================================================================
# PART 2: 数据集类 
# ==============================================================================
class TerraHybridCLIPDataset(Dataset):
    def __init__(self, dir_list, transform=None, max_length=77, is_train=False):
        self.transform = transform
        self.max_length = max_length
        self.samples = [] 
        self.class_counts = Counter() 
        self.is_train = is_train
        self.dataset_type = "Train/Valid (Raw)" if is_train else "Test"
        
        # 动态决定加载哪个 JSON 文件
        self.target_json = "annotations.json" if is_train else "local_label.json"
        
        print(f"\n📂 [{self.dataset_type} Loader] 扫描 {len(dir_list)} 个路径, 目标文件: {self.target_json}")
        
        for folder_path in dir_list:
            if not os.path.exists(folder_path):
                print(f"   ❌ [跳过] 路径不存在: {folder_path}")
                continue
            
            json_path = os.path.join(folder_path, self.target_json)
            if os.path.exists(json_path):
                if self.is_train:
                    self._load_train_annotations(folder_path, json_path)
                else:
                    self._load_test_labels(folder_path, json_path)
            else:
                print(f"   ❌ [跳过] 未找到 {self.target_json}: {folder_path}")

        covered_classes = len(self.class_counts.keys())
        print(f"\n   📊 [{self.dataset_type}] 最终统计:")
        print(f"      - 总图片数: {len(self.samples)} 张")
        print(f"      - 覆盖类别: 全局 {len(ALL_TERRAIN_CLASSES)} 类中的 {covered_classes} 类。")
        
        if covered_classes > 0:
            print(f"      - 类别明细 (降序):")
            for label, count in self.class_counts.most_common():
                print(f"        * {label}: {count} 张")

    def _load_train_annotations(self, folder_path, json_path):
        """专门用于训练阶段：读取 annotations.json，提取多粒度属性"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            valid_count = 0
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
                        "fine_text": fine_text
                    })
                    self.class_counts[material] += 1
                    valid_count += 1
            print(f"      -> {json_path.split('/')[-3]}: 成功加载 {valid_count} 张 (Train Annotation)")
        except Exception as e:
            print(f"      -> [Error] 解析 {json_path} 失败: {e}")

    def _load_test_labels(self, folder_path, json_path):
        """专门用于测试/验证阶段：读取 local_label.json，提取绝对真值"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 适配 local_label.json 的 {"items": [...]} 格式
            items = data.get("items", [])
            valid_count = 0
            
            for item in items:
                img_name = item.get("name")
                label_str = item.get("material", "").lower()
                
                if not img_name or not label_str: continue
                if label_str in ALIAS_MAP: label_str = ALIAS_MAP[label_str]
                
                if label_str in GLOBAL_LABEL_MAP:
                    full_img_path = os.path.join(folder_path, img_name)
                    if not os.path.exists(full_img_path): continue
                        
                    self.samples.append({
                        "image": full_img_path,
                        "label_idx": GLOBAL_LABEL_MAP[label_str],
                        # 测试阶段不需要文本生成，留空即可节省内存
                        "coarse_text": "",
                        "fine_text": ""
                    })
                    self.class_counts[label_str] += 1
                    valid_count += 1
            print(f"      -> {json_path.split('/')[-3]}: 成功加载 {valid_count} 张 (GT Label)")
        except Exception as e:
            print(f"      -> [Error] 解析 {json_path} 失败: {e}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        try:
            image = Image.open(item['image']).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224))
        if self.transform:
            image = self.transform(image)
            
        coarse_text = item['coarse_text'][:self.max_length]
        fine_text = item['fine_text'][:self.max_length]
        return image, coarse_text, fine_text, item['label_idx']

# ==========================================
# 新增: Dataset 包装器，适配 TerraCLIP 的四元组返回格式
# ==========================================
class DatasetWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        # 接收底层 dataset 返回的 4 个元素
        image, coarse_text, fine_text, label_idx = self.subset[index]
        
        # 独立应用 transform
        if self.transform:
            image = self.transform(image)
            
        return image, coarse_text, fine_text, label_idx

    def __len__(self):
        return len(self.subset)


# ==============================================================================
# PART 3: 训练与验证核心逻辑
# ==============================================================================
def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training") 
    
    for images, coarse_texts, fine_texts, labels in pbar:
        images = images.to(device)
        coarse_tokens = clip.tokenize(coarse_texts, truncate=True).to(device)
        fine_tokens = clip.tokenize(fine_texts, truncate=True).to(device)
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            image_features = model.encode_image(images)
            coarse_features = model.encode_text(coarse_tokens)
            fine_features = model.encode_text(fine_tokens)
            
            image_features = F.normalize(image_features, dim=-1)
            coarse_features = F.normalize(coarse_features, dim=-1)
            fine_features = F.normalize(fine_features, dim=-1)
            
            logit_scale = model.logit_scale.exp()
            
            logits_img_coarse = logit_scale * image_features @ coarse_features.t()
            logits_coarse_img = logits_img_coarse.t()
            
            logits_img_fine = logit_scale * image_features @ fine_features.t()
            logits_fine_img = logits_img_fine.t()
            
            labels_idx = torch.arange(len(images), device=device)
            
            loss_coarse = (F.cross_entropy(logits_img_coarse, labels_idx) + 
                           F.cross_entropy(logits_coarse_img, labels_idx)) / 2
            
            loss_fine = (F.cross_entropy(logits_img_fine, labels_idx) + 
                         F.cross_entropy(logits_fine_img, labels_idx)) / 2
            
            loss = CONFIG["coarse_weight"] * loss_coarse + (1 - CONFIG["coarse_weight"]) * loss_fine

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}") 
        wandb.log({"train_loss": loss.item()})
        
    return total_loss / len(loader)

def evaluate(model, loader, device, all_labels):
    model.eval()
    print("\nComputing multi-granularity text prototypes for Inference (Aligned with Paper)...")
    
    # ==========================================
    # 1. 粗粒度文本 (Coarse-grained prompts)
    # ==========================================
    coarse_prompts = [f"The terrain material is {label}." for label in all_labels]
    coarse_tokens = clip.tokenize(coarse_prompts, truncate=True).to(device)
    
    # ==========================================
    # 2. 细粒度属性池 (Fine-grained attributes)
    # ==========================================
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
    
    # 将细粒度提示词展平为 1D 列表
    for label in all_labels:
        for attr_key, attr_vals in fine_attributes.items():
            for val in attr_vals:
                fine_prompts_flat.append(f"The terrain material is {label}, characterized by {attr_key} being {val}.")
                
    # 每个类别拥有的属性描述数量
    M = len(fine_prompts_flat) // K  
    
    with torch.no_grad():
        coarse_features = model.encode_text(coarse_tokens)
        coarse_features = F.normalize(coarse_features, dim=-1)
        
        fine_tokens = clip.tokenize(fine_prompts_flat, truncate=True).to(device)
        fine_features_flat = model.encode_text(fine_tokens)
        fine_features_flat = F.normalize(fine_features_flat, dim=-1)
        
    all_preds = []
    all_targets = []
    
    # 显著性增强因子 / 多样性权重
    lambda_weight = 0.3 
    
    with torch.no_grad():
        logit_scale = model.logit_scale.exp().item()
        
        for images, _, _, labels in tqdm(loader, desc="Validating (Dynamic Weighting)", leave=False):
            images = images.to(device)
            
            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            
            # -----------------------------------------------------------------
            # 步骤 1: 粗粒度概率计算
            # -----------------------------------------------------------------
            logits_coarse = logit_scale * (image_features @ coarse_features.t()) # [B, K]
            probs_coarse = F.softmax(logits_coarse, dim=-1) # [B, K]
            
            # -----------------------------------------------------------------
            # 步骤 2: 细粒度概率计算与聚合
            # -----------------------------------------------------------------
            logits_fine = logit_scale * (image_features @ fine_features_flat.t()) # [B, K*M]
            probs_fine_flat = F.softmax(logits_fine, dim=-1) # [B, K*M]
            
            # 重塑为 [Batch, K_classes, M_attributes]
            probs_fine = probs_fine_flat.view(-1, K, M)
            
            # 聚合分数: Sum + lambda * Max
            sum_score = probs_fine.sum(dim=-1) # [B, K]
            max_score = probs_fine.max(dim=-1).values # [B, K]
            fine_score = sum_score + lambda_weight * max_score # [B, K]
            
            # -----------------------------------------------------------------
            # 步骤 3: 动态熵权重计算 (Alpha)
            # -----------------------------------------------------------------
            import numpy as np # 确保能调用 np.log
            # 计算香农熵
            entropy = -torch.sum(probs_coarse * torch.log(probs_coarse + 1e-9), dim=-1) # [B]
            
            # 归一化熵作为权重 (熵越大，置信度越低，alpha 越大)
            alpha = entropy / np.log(K) 
            alpha = alpha.unsqueeze(-1) # 扩展维度以支持广播 [B, 1]
            
            # -----------------------------------------------------------------
            # 步骤 4: 最终置信度融合预测
            # -----------------------------------------------------------------
            final_scores = (1.0 - alpha) * probs_coarse + alpha * fine_score # [B, K]
            preds = final_scores.argmax(dim=-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    return acc, all_targets, all_preds


# ==============================================================================
# PART 4: 主函数
# ==============================================================================
def main(args):
    if args.exp_name not in EXPERIMENTS:
        raise ValueError(f"❌ 实验名称 '{args.exp_name}' 未在 EXPERIMENTS 中定义！试试填 benchmark3")
    
    exp_cfg = EXPERIMENTS[args.exp_name]
    CONFIG["train_dirs"] = exp_cfg["train_dirs"]
    # 提取多测试集套件
    CONFIG["test_suites"] = exp_cfg["test_suites"] 
    
    exp_save_dir = os.path.join(CONFIG["save_dir"], args.exp_name)
    os.makedirs(exp_save_dir, exist_ok=True)
    
    wandb.init(project=CONFIG["wandb_project"], config=CONFIG, name=f"run_{args.exp_name}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=======================================================")
    print(f"🚀 启动实验: {args.exp_name} | 使用设备: GPU {args.gpu}")
    print(f"📁 模型保存至: {exp_save_dir}")
    print(f"=======================================================\n")
    
    model, preprocess = clip.load(CONFIG["model_name"], device=device)
    # 🚨 核心修复：强制转回 FP32，防止 GradScaler 缩放 FP16 梯度崩溃！
    model = model.float()
    
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)

    train_transform_safe = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])), 
        
        # 🚨 关键修复：把自定义的 imgaug 增强模块加进来！放在 Resize 之后以保证速度
        ImgAugTransform(prob=0.5), 
        
        transforms.RandomHorizontalFlip(),
        # 💡 注意：我已经帮你把 ColorJitter 注释掉了，因为 ImgAugTransform 里面已经包含了更丰富的亮度/对比度/饱和度变化，叠加使用会导致图片损坏严重。
        # transforms.ColorJitter(brightness=0.2, contrast=0.2), 
        transforms.ToTensor(),
        transforms.Normalize(clip_mean, clip_std)
    ])
    
    valid_transform_safe = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(clip_mean, clip_std)
    ])

    if CONFIG["fine_tune_method"] == "lora":
        print(f"Initializing LoRA (Rank={CONFIG['lora_rank']})...")
        config = LoraConfig(
            r=CONFIG["lora_rank"],
            lora_alpha=CONFIG["lora_alpha"],
            target_modules=[ "c_fc", "c_proj"],
            lora_dropout=CONFIG["lora_dropout"],
            bias="none"
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    
    # ------------------ 新的数据集划分逻辑 ------------------
    # 1. 读取含有文本真值的完整训练集 (is_train=True, 暂不挂载 transform)
    full_train_dataset = TerraHybridCLIPDataset(CONFIG["train_dirs"], transform=None, is_train=True)
    
    # 2. 80/20 切分
    total_size = len(full_train_dataset)
    train_size = int(0.8 * total_size)
    valid_size = total_size - train_size
    raw_train_set, raw_valid_set = random_split(full_train_dataset, [train_size, valid_size])
    
    # 3. 通过 Wrapper 挂载变换
    real_train_dataset = DatasetWrapper(raw_train_set, transform=train_transform_safe)
    real_valid_dataset = DatasetWrapper(raw_valid_set, transform=valid_transform_safe)

    print(f"\n📐 数据集划分完毕:")
    print(f"   - 训练集 (Train): {len(real_train_dataset)} 张")
    print(f"   - 验证集 (Valid): {len(real_valid_dataset)} 张 (用于 Zero-Shot 评估选模型)")

    # 4. 生成 Loader
    train_loader = DataLoader(real_train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)
    valid_loader = DataLoader(real_valid_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
    # --------------------------------------------------------
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')
    
    best_acc = 0.0
    patience = 5
    counter = 0 
    patience_min_delta = 0.001
    
    best_model_dir = os.path.join(exp_save_dir, "clip_hybrid_best")
    
    print(f"\n🚀 开始训练 (Epochs: {CONFIG['epochs']}) ...")
    
    for epoch in range(CONFIG["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_acc, _, _ = evaluate(model, valid_loader, device, ALL_TERRAIN_CLASSES)
        
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Train Loss: {train_loss:.4f} | Val Acc (Zero-Shot): {val_acc:.4f}")
        wandb.log({"val_acc": val_acc, "epoch": epoch})
        
        if val_acc > best_acc + patience_min_delta:
            best_acc = val_acc
            counter = 0 
            model.save_pretrained(best_model_dir)
            print(f">>> 💾 Best Model Saved to {best_model_dir}!")
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
    if os.path.exists(best_model_dir):
        from peft import PeftModel
        base_model, _ = clip.load(CONFIG["model_name"], device=device)
        base_model = base_model.float() # 加载测试也要同样转回 FP32
        model = PeftModel.from_pretrained(base_model, best_model_dir)
        model.to(device)
        print("✅ 已成功加载最佳模型")
    else:
        print("⚠️ 未找到最佳模型目录，使用当前模型进行测试")
    
    # 存储所有天气的最终成绩
    benchmark_results = {}

    for condition_name, test_dirs in CONFIG["test_suites"].items():
        print(f"\n[Testing] 正在测试环境: {condition_name}")
        
        # 动态加载当前天气的 Dataset 和 DataLoader
        test_dataset = TerraHybridCLIPDataset(test_dirs, transform=valid_transform_safe, is_train=False)
        test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
        
        # 🚨 极简调用 evaluate 函数
        final_acc, targets, preds = evaluate(model, test_loader, device, ALL_TERRAIN_CLASSES)
        benchmark_results[condition_name] = final_acc
        
        print(f">>> {condition_name} Test Accuracy: {final_acc:.4f}")
        
        unique_classes = sorted(list(set(targets)))
        target_names = [ALL_TERRAIN_CLASSES[i] for i in unique_classes]
        print(classification_report(targets, preds, labels=unique_classes, target_names=target_names, digits=4, zero_division=0))
    
    # =========================================================================
    # 打印并保存最终漂亮的汇总表格数据
    # =========================================================================
    print("\n=======================================================")
    print("📊 Benchmark 3: Robust Classification Final Results (TerraCLIP)")
    print("=======================================================")
    
    # 1. 终端打印
    for condition, acc in benchmark_results.items():
        print(f"| {condition.ljust(10)} | {acc:.4f} |")
    print("=======================================================\n")

    # 2. 自动保存为 CSV 文件
    results_csv_path = os.path.join(exp_save_dir, f"{args.exp_name}_results.csv")
    
    with open(results_csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Test Condition", "Accuracy"])
        for condition, acc in benchmark_results.items():
            writer.writerow([condition, f"{acc:.4f}"])
            
    print(f"📁 汇总结果已成功保存至: {results_csv_path}")
    
    wandb.finish()


if __name__ == "__main__":
    main(args)