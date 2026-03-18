import os
# os.environ["WANDB_MODE"] = "offline"
import argparse


# ==============================================================================
# 🚨 极其重要：必须在导入 torch 前完成参数解析和 GPU 环境变量设置！
# ==============================================================================
parser = argparse.ArgumentParser(description="多卡并发训练 TerraCLIP")
parser.add_argument("--exp_name", type=str, required=True, help="实验名称，如 exp1, exp2")
parser.add_argument("--gpu", type=str, default="0", help="指定要使用的 GPU ID，如 0 或 1")
args, _ = parser.parse_known_args()

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
import pandas as pd

# ==============================================================================
# PART 0: 实验集配置
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

# ==============================================================================
# PART 1: 全局配置与标签定义 [修改点：二分类]
# ==============================================================================
TRAVERSABILITY_CLASSES = ["non-traversable", "traversable"]
GLOBAL_LABEL_MAP = {label: idx for idx, label in enumerate(TRAVERSABILITY_CLASSES)}

CONFIG = {
    "batch_size": 512,          
    "lr": 5e-5,                 
    "epochs": 50,
    "num_workers": 8,
    "img_size": 224,            
    "coarse_weight": 0.6,       
    "model_name": "ViT-B/32",
    "fine_tune_method": "lora", 
    "save_dir": "./clip_checkpoints/benchmark6",
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "wandb_project": "TerraVerse-TerraCLIP-Benchmark6",
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
        print(f"      - 覆盖类别: 全局 {len(TRAVERSABILITY_CLASSES)} 类中的 {covered_classes} 类。")
        
        if covered_classes > 0:
            print(f"      - 类别明细 (降序):")
            for label, count in self.class_counts.most_common():
                print(f"        * {label}: {count} 张")

    def _load_train_annotations(self, folder_path, json_path):
        """[修改点]：适配二分类文本提示词"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            valid_count = 0
            for item in data:
                img_name = item.get("name")
                final_ann = item.get("final_annotation", {})
                
                # 读取关键的 traversability_hint
                vis_hints = final_ann.get("visual_physical_hints", {})
                trav_hint = vis_hints.get("traversability_hint", "").lower()
                
                if not img_name or not trav_hint: continue
                
                if trav_hint in GLOBAL_LABEL_MAP:
                    full_img_path = os.path.join(folder_path, img_name)
                    if not os.path.exists(full_img_path): continue
                        
                    material = final_ann.get("material", "unknown")
                    global_context = final_ann.get("global_context", {})
                    vis_attrs = final_ann.get("visual_attributes", {})
                    
                    weather = global_context.get("weather", "unknown")
                    lighting = global_context.get("lighting", "unknown")
                    smoothness = vis_attrs.get("smoothness", "unknown")
                    moisture = vis_attrs.get("moisture", "unknown")
                    fric_hint = vis_hints.get("friction_hint", "unknown")

                    # 应用新的 Prompt 格式
                    coarse_text = f"The traversability of material is {trav_hint}."
                    fine_text = (f"The traversability of material is {trav_hint}, "
                                 f"The terrain material is {material}, "
                                 f"weather is {weather}, lighting is {lighting}, "
                                 f"smoothness is {smoothness}, moisture is {moisture}, "
                                 f"friction_hint is {fric_hint}.")
                    
                    self.samples.append({
                        "image": full_img_path,
                        "label_idx": GLOBAL_LABEL_MAP[trav_hint],
                        "coarse_text": coarse_text,
                        "fine_text": fine_text
                    })
                    self.class_counts[trav_hint] += 1
                    valid_count += 1
            print(f"      -> {json_path.split('/')[-3]}: 成功加载 {valid_count} 张 (Train Annotation)")
        except Exception as e:
            print(f"      -> [Error] 解析 {json_path} 失败: {e}")

    def _load_test_labels(self, folder_path, json_path):
        """[修改点]：读取 local_label.json 的 traversability"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            items = data.get("items", [])
            valid_count = 0
            
            for item in items:
                img_name = item.get("name")
                # 改为读取 traversability
                trav_str = item.get("traversability", "").lower()
                
                if not img_name or not trav_str: continue
                
                if trav_str in GLOBAL_LABEL_MAP:
                    full_img_path = os.path.join(folder_path, img_name)
                    if not os.path.exists(full_img_path): continue
                        
                    self.samples.append({
                        "image": full_img_path,
                        "label_idx": GLOBAL_LABEL_MAP[trav_str],
                        "coarse_text": "",
                        "fine_text": ""
                    })
                    self.class_counts[trav_str] += 1
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
# Dataset 包装器
# ==========================================
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
    print("\nComputing multi-granularity text prototypes for Inference...")
    
    # ==========================================
    # 1. 粗粒度文本 [修改点：适配新提示词]
    # ==========================================
    coarse_prompts = [f"The traversability of material is {label}." for label in all_labels]
    coarse_tokens = clip.tokenize(coarse_prompts, truncate=True).to(device)
    
    # ==========================================
    # 2. 细粒度属性池 [修改点：加入 material 作为属性]
    # ==========================================
    fine_attributes = {
        "material": ["asphalt", "brick", "cobble", "concrete", "tile", "coated floor", "flagstone", "board", "dirt", "gravel", "mud", "mulch", "rock", "sand", "soil", "puddle", "snow", "water", "ice", "moss", "grass", "leaves"],
        "weather": ["sunny", "cloudy", "rainy", "foggy", "snowy", "unknown"],
        "lighting": ["strong sunlight", "low light", "shadowed", "dark"],
        "moisture": ["dry", "moist", "wet"],
        "smoothness": ["smooth", "slightly uneven", "severely uneven"],
        "friction_hint": ["high", "medium", "low"]
    }
    
    K = len(all_labels)
    fine_prompts_flat = []
    
    for label in all_labels:
        for attr_key, attr_vals in fine_attributes.items():
            for val in attr_vals:
                fine_prompts_flat.append(f"The traversability of material is {label}, characterized by {attr_key} being {val}.")
                
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
    
    with torch.no_grad():
        logit_scale = model.logit_scale.exp().item()
        
        for images, _, _, labels in tqdm(loader, desc="Validating (Dynamic Weighting)", leave=False):
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
            
            import numpy as np 
            entropy = -torch.sum(probs_coarse * torch.log(probs_coarse + 1e-9), dim=-1) 
            
            alpha = entropy / np.log(K) 
            alpha = alpha.unsqueeze(-1) 
            
            final_scores = (1.0 - alpha) * probs_coarse + alpha * fine_score 
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
        raise ValueError(f"❌ 实验名称 '{args.exp_name}' 未在 EXPERIMENTS 中定义！")
    
    exp_cfg = EXPERIMENTS[args.exp_name]
    CONFIG["train_dirs"] = exp_cfg["train_dirs"]
    CONFIG["test_dirs"] = exp_cfg["test_dirs"]
    
    exp_save_dir = os.path.join(CONFIG["save_dir"], args.exp_name)
    os.makedirs(exp_save_dir, exist_ok=True)
    
    wandb.init(project=CONFIG["wandb_project"], config=CONFIG, name=f"run_trav_{args.exp_name}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=======================================================")
    print(f"🚀 启动实验: {args.exp_name} | 使用设备: GPU {args.gpu}")
    print(f"📁 模型保存至: {exp_save_dir}")
    print(f"=======================================================\n")
    
    model, preprocess = clip.load(CONFIG["model_name"], device=device)
    model = model.float()
    
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
    
    full_train_dataset = TerraHybridCLIPDataset(CONFIG["train_dirs"], transform=None, is_train=True)
    
    total_size = len(full_train_dataset)
    train_size = int(0.8 * total_size)
    valid_size = total_size - train_size
    raw_train_set, raw_valid_set = random_split(full_train_dataset, 
                                                [train_size, valid_size],
                                                generator=torch.Generator().manual_seed(42))
    
    real_train_dataset = DatasetWrapper(raw_train_set, transform=train_transform_safe)
    real_valid_dataset = DatasetWrapper(raw_valid_set, transform=valid_transform_safe)
    
    test_dataset = TerraHybridCLIPDataset(CONFIG["test_dirs"], transform=valid_transform_safe, is_train=False)

    print(f"\n📐 数据集划分完毕:")
    print(f"   - 训练集 (Train): {len(real_train_dataset)} 张")
    print(f"   - 验证集 (Valid): {len(real_valid_dataset)} 张")
    print(f"   - 测试集 (Test) : {len(test_dataset)} 张")

    train_loader = DataLoader(real_train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)
    valid_loader = DataLoader(real_valid_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')
    
    best_acc = 0.0
    patience = 5
    counter = 0 
    patience_min_delta = 0.001
    
    # [修改点]：更新保存名以区分任务
    best_model_dir = os.path.join(exp_save_dir, "clip_traversable_best")
    
    print(f"\n🚀 开始训练 (Epochs: {CONFIG['epochs']}) ...")
    
    for epoch in range(CONFIG["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        # [修改点]：传入新的二分类类别列表
        val_acc, _, _ = evaluate(model, valid_loader, device, TRAVERSABILITY_CLASSES)
        
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
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

    print("\n=== 🏆 训练结束，加载最佳模型进行最终盲测 (Test) ===")
    if os.path.exists(best_model_dir):
        from peft import PeftModel
        base_model, _ = clip.load(CONFIG["model_name"], device=device)
        base_model = base_model.float() 
        model = PeftModel.from_pretrained(base_model, best_model_dir)
        model.to(device)
        print("✅ 已成功加载最佳模型")
    else:
        print("⚠️ 未找到最佳模型目录，使用当前模型进行测试")
    
    final_acc, targets, preds = evaluate(model, test_loader, device, TRAVERSABILITY_CLASSES)
    print(f"Final Test Accuracy: {final_acc:.4f}")
    
    unique_classes = sorted(list(set(targets)))
    target_names = [TRAVERSABILITY_CLASSES[i] for i in unique_classes]
    print(classification_report(targets, preds, labels=unique_classes, target_names=target_names, digits=4, zero_division=0))
    
    report_dict = classification_report(targets, preds, labels=unique_classes, target_names=target_names, digits=4, zero_division=0, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    
    df_report.loc['Final_Test_Accuracy'] = [final_acc, None, None, None]
    
    # [修改点]：输出文件名适配
    csv_filename = f"{args.exp_name}_traversable_test_results.csv"
    csv_path = os.path.join(exp_save_dir, csv_filename)
    df_report.to_csv(csv_path, index=True)
    
    print(f"\n📊 [保存成功] 测试结果详细指标已导出至: {csv_path}")
    
    wandb.finish()

if __name__ == "__main__":
    main(args)