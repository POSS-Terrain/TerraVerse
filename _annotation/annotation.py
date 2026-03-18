import json
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
from dataclasses import dataclass, asdict, field
import argparse
import sys

from openai import OpenAI
from tqdm import tqdm
import cv2
import os


# ============================================================
# API 统计类
# ============================================================

@dataclass
class APIStats:
    """记录 API 调用的统计信息"""
    generator_calls: int = 0
    generator_success: int = 0
    generator_failed: int = 0
    reviewer_calls: int = 0
    reviewer_success: int = 0
    reviewer_failed: int = 0
    
    def record_generator_call(self, success: bool):
        """记录一次 Generator API 调用"""
        self.generator_calls += 1
        if success:
            self.generator_success += 1
        else:
            self.generator_failed += 1
    
    def record_reviewer_call(self, success: bool):
        """记录一次 Reviewer API 调用"""
        self.reviewer_calls += 1
        if success:
            self.reviewer_success += 1
        else:
            self.reviewer_failed += 1
    
    @property
    def generator_success_rate(self) -> float:
        """Generator 成功率"""
        if self.generator_calls == 0:
            return 0.0
        return self.generator_success / self.generator_calls
    
    @property
    def reviewer_success_rate(self) -> float:
        """Reviewer 成功率"""
        if self.reviewer_calls == 0:
            return 0.0
        return self.reviewer_success / self.reviewer_calls
    
    @property
    def total_calls(self) -> int:
        """总 API 调用次数"""
        return self.generator_calls + self.reviewer_calls
    
    @property
    def total_success(self) -> int:
        """总成功次数"""
        return self.generator_success + self.reviewer_success
    
    @property
    def overall_success_rate(self) -> float:
        """总体成功率"""
        if self.total_calls == 0:
            return 0.0
        return self.total_success / self.total_calls
    
    def to_dict(self) -> Dict:
        """转换为字典格式，包含计算的指标"""
        return {
            "generator_calls": self.generator_calls,
            "generator_success": self.generator_success,
            "generator_failed": self.generator_failed,
            "generator_success_rate": round(self.generator_success_rate, 4),
            "reviewer_calls": self.reviewer_calls,
            "reviewer_success": self.reviewer_success,
            "reviewer_failed": self.reviewer_failed,
            "reviewer_success_rate": round(self.reviewer_success_rate, 4),
            "total_calls": self.total_calls,
            "total_success": self.total_success,
            "overall_success_rate": round(self.overall_success_rate, 4),
        }

# 配置区
# ============================================================

# DATASET_ROOT = Path("./TerraData/DeepScene")
# OUTPUT_DIR = Path("./Annotation/outputs")
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ANNOTATION_DIR = Path("./Annotation/outputs")
# ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

GENERATOR_MODEL = "qwen3-vl-flash"
REVIEWER_MODEL = "gpt-5-mini"

CONF_ACCEPT = 0.85
CONF_WEAK = 0.6

GLOBAL_IMAGE_WIDTH = 224  # Resize 后的全景图宽度

# ============================================================
# Prompt 设计
# ============================================================

SYSTEM_PROMPT_GENERATOR = """
You are a deterministic terrain annotation engine used for academic dataset construction.

Your purpose is to generate consistent, conservative, and strictly constrained annotations, and estimate annotation confidence carefully.
You must base your decisions ONLY on the provided images.

Rules:
- Focus on terrain surface properties only.
- Do NOT infer unobservable physical measurements.
- Use ONLY the predefined label sets.
- Output MUST be valid JSON with no extra text.
""".strip()

# {
#       "name": "img_000001_0001.png",
#       "label": "road",
#       "top_left": [
#         112,
#         672
#       ],
#       "bottom_right": [
#         335,
#         895
#       ],
#       "category": "Structured / Artificial Surfaces",
#       "material": null,
#       "function": "road",
#       "traversability": null
#     },

def build_user_prompt(existing_annotations: dict) -> str:
    """
    Build a system-level surface annotation prompt for academic dataset labeling.
    
    Args:
        label_name (str): Ground truth category label.
        
    Returns:
        str: Fully constructed prompt text.
    """

    # 对于已有的真值标注，有什么给什么，三个小部分加起来
    input_description_1 = f"""
        INPUT: 

        You are given:

        - Image 1: A full panoramic image providing global environmental context, with a red box indicating the location of the cropped patch (Image 2).
        - Image 2: A cropped local patch derived from Image 1."""
    
    # - Ground truth category label: "{label_name}" 
    input_description_2 = f"""
        - Existing partial annotations (authoritative when not NULL): {existing_annotations}"""

    input_description_3 = f"""
        The cropped patch contains approximately 95% of one terrain category".

        The material field in the output MUST be consistent with the provided ground truth label ('{existing_annotations.get("material", "NULL")}') if it is not NULL.
        Do not contradict the ground truth category.
        """
    
    input_description = input_description_1 + input_description_2 + input_description_3

    annotation_objective = """
        ANNOTATION OBJECTIVE:

        Produce a structured annotation describing:

        1. Global environmental context (from Image 1)
        2. Local surface characteristics (from Image 2)
        3. Visual physical property hints based strictly on visible evidence

        All decisions must prioritize visual evidence over assumptions.
        """

    annotation_procedure = """
        ANNOTATION PROCEDURE: 

        Step 1 — Global Context (Image 1 only)

        Determine:
        - Weather condition
        - Lighting condition

        Step 2 — Local Surface Analysis (Image 2 only)

        - Identify dominant visible surface texture.
        - Ignore minor objects, noise, borders, or annotation artifacts.
        - Focus only on the primary surface material.

        Step 3 — Attribute Assignment (Patch-Level)

        Assign:
        - Surface smoothness (visual estimate)
        - Local surface moisture (based only on visible signs)

        Step 4 — Visual Physical Hints

        Based strictly on visible cues (NOT world knowledge), estimate:

        - Friction level
        - Traversability

        Use conservative judgment.
        Avoid extreme values unless strongly supported by visual evidence.

        Step 5 — Internal Consistency and Confidence Verification

        Before producing the final output, perform an internal reliability check.

        Evaluate confidence based on:

        A) Image Quality

        - Is the patch visually clear, well lit, and textures distinguishable?

        B) Material Clarity

        - Do visual cues strongly support a single material,
        or could multiple materials plausibly match?

        C) Physical Consistency

        - Ensure material, smoothness, moisture, and traversability are logically aligned.
        Example:
        mud + dry + high traversability → inconsistent
        asphalt + smooth + high traversability → consistent

        D) Label Consistency

        - Predicted material must agree with provided category or locked annotations.

        Rules:

        - Confidence should decrease when ambiguity or visual uncertainty exists.
        - The weakest factor should dominate the final confidence judgement.
        - If visual evidence is unclear, prefer conservative attribute choices.

        Do NOT output reasoning.
        """

    # 指定所有可能的标签
    allowed_values = """
        ALLOWED ATTRIBUTE VALUES (STRICT ENUMS): 

        Weather:
        ["sunny", "cloudy", "rainy", "foggy", "snowy", "unknown"]

        Lighting:
        ["strong sunlight", "low light", "shadowed", "dark"]

        Material:
        ["asphalt", "brick", "cobble", "concrete", "tile", "coated floor", "flagstone", "board", 
            "dirt", "gravel", "mud", "mulch", "sand", "soil", "puddle", "snow", "water", "ice", "rock"
            "moss", "grass", "leaves"]

        Local Surface Moisture:
        ["dry", "moist", "wet"]

        Surface Smoothness:
        ["smooth", "slightly Uneven", "severely Uneven"]

        Visual Friction Hint:
        ["high", "medium", "low"]

        Visual Traversability Hint:
        ["traversable", "non-traversable"]
        """

    output_constraints = """
        OUTPUT REQUIREMENTS:

        - Output STRICT JSON only.
        - No markdown formatting.
        - No explanations.
        - No additional keys.
        - No missing keys.
        - No null values.
        - All categorical fields MUST use one of the allowed values exactly.
        - Output must be syntactically valid JSON.

        ==================================================
        OUTPUT FORMAT
        ==================================================

        {
            "global_context": {
                "weather": "",
                "lighting": "",
            },
            "material": "",
            "visual_attributes": {
                "smoothness": "",
                "moisture": ""
            },
            "visual_physical_hints": {
                "friction_hint": "",
                "traversability_hint": ""
            },
            "confidence_breakdown": {
                "image_quality_score": 0-1,
                "material_clarity_score": 0-1,
                "physical_consistency_score": 0-1,
                "label_match_score": 0-1
            },
        }
        """

    full_prompt = (
        input_description
        + annotation_objective
        + annotation_procedure
        + allowed_values
        + output_constraints
    )

    return full_prompt

SYSTEM_PROMPT_REVIEWER = """
You are a strict annotation verifier for an off-road terrain dataset.

Your task is to verify whether a given annotation is visually supported and internally consistent.

Rules:
- Judge ONLY based on the images.
- Do NOT introduce new attributes.
- Be conservative.
- Output JSON only.
""".strip()


# ============================================================
# 工具函数
# ============================================================

def encode_image(img: Path) -> str:
    """支持传入文件路径 (`Path`/`str`) 或 OpenCV `ndarray`，返回 base64 编码字符串。"""
    # 如果是文件路径或类路径对象
    if isinstance(img, (str, os.PathLike, Path)):
        with open(img, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # 否则假设是 OpenCV 的 ndarray，使用 imencode 编码为 PNG 再转 base64
    try:
        success, buf = cv2.imencode('.png', img)
        if not success:
            raise ValueError("cv2.imencode failed")
        return base64.b64encode(buf.tobytes()).decode('utf-8')
    except Exception:
        raise


def draw_bbox(global_img: Path, tl: List[int], br: List[int]):
    img = cv2.imread(str(global_img))
    cv2.rectangle(img, tuple(tl), tuple(br), (0, 0, 255), 6)
    # cv2.imwrite(str(out), img)
    return img


# ============================================================
# Generator
# ============================================================

def call_generator(
    client: OpenAI,
    existing_annotations: dict,
    global_img: Path,
    local_img: Path,
    annotation_model: str = GENERATOR_MODEL,
    api_stats: Optional[APIStats] = None,
) -> Optional[Dict]:

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_GENERATOR},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_user_prompt(existing_annotations)},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{encode_image(global_img)}",
                    "detail": "high"}},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{encode_image(local_img)}",
                    "detail": "high"}}
            ]
        }
    ]

    try:
        resp = client.chat.completions.create(
            model=annotation_model,
            messages=messages,
            # temperature=0.2,
            response_format={"type": "json_object"}
        )
        result = json.loads(resp.choices[0].message.content)
        if api_stats:
            api_stats.record_generator_call(True)
        return result
    except Exception as e:
        print("[Generator failed]", e)
        if api_stats:
            api_stats.record_generator_call(False)
        return None

# ============================================================
def check_annotation(ann: Dict, existing_annotations: dict) -> bool:
    """检查返回的标注是否符合要求"""
    try:
        # 检查顶层字段
        assert "global_context" in ann
        assert "material" in ann
        assert "visual_attributes" in ann
        assert "visual_physical_hints" in ann

        # 检查 global_context 字段
        gc = ann["global_context"]
        assert "weather" in gc
        assert "lighting" in gc

        # 检查 visual_attributes 字段
        va = ann["visual_attributes"]
        assert "smoothness" in va
        assert "moisture" in va

        # 检查 visual_physical_hints 字段
        vph = ann["visual_physical_hints"]
        assert "friction_hint" in vph
        assert "traversability_hint" in vph

        # 检查 material 是否与 existing_annotations 一致（如果有的话）
        if existing_annotations.get("material") is not None:
            assert ann["material"] == existing_annotations["material"]
        # 检查 traversability 是否与 existing_annotations 一致（如果有的话）
        if existing_annotations.get("traversability") is not None:
            assert ann["visual_physical_hints"]["traversability_hint"] == existing_annotations["traversability"]

        return True
    except AssertionError:
        return False


def get_annotation_confidence(ann: Dict) -> float:
    """从标注中提取综合置信度"""
    try:
        if "confidence_breakdown" not in ann:
            return 0.0
        
        cb = ann["confidence_breakdown"]
        # 计算四个置信度分数的平均值
        scores = [
            cb.get("image_quality_score", 0),
            cb.get("material_clarity_score", 0),
            cb.get("physical_consistency_score", 0),
            cb.get("label_match_score", 0),
        ]
        return sum(scores) / len(scores) if scores else 0.0
    except Exception:
        return 0.0

# ============================================================
# Reviewer
# ============================================================

# def call_reviewer(
#     client: OpenAI,
#     annotation: Dict,
#     global_img: Path,
#     local_img: Path,
#     api_stats: Optional[APIStats] = None
# ) -> bool:

#     user_prompt = f"""
#         Annotation:
#         {json.dumps(annotation, indent=2)}

#         Question:
#         Does this annotation accurately and consistently describe the terrain?

#         Answer JSON only:
#         {{"verdict": "Yes" or "No"}}
#         """.strip()

#     messages = [
#         {"role": "system", "content": SYSTEM_PROMPT_REVIEWER},
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": user_prompt},
#                 {"type": "image_url", "image_url": {
#                     "url": f"data:image/png;base64,{encode_image(global_img)}"}},
#                 {"type": "image_url", "image_url": {
#                     "url": f"data:image/png;base64,{encode_image(local_img)}"}}
#             ]
#         }
#     ]

#     try:
#         resp = client.chat.completions.create(
#             model=REVIEWER_MODEL,
#             messages=messages,
#             temperature=0.0,
#             response_format={"type": "json_object"}
#         )
#         result = json.loads(resp.choices[0].message.content)["verdict"] == "Yes"
#         if api_stats:
#             api_stats.record_reviewer_call(True)
#         return result
#     except Exception:
#         if api_stats:
#             api_stats.record_reviewer_call(False)
#         return False


# ============================================================
# 一致性指标
# ============================================================

# def self_consistency(anns: List[Dict], field: Tuple[str, ...]) -> float:
#     """计算多个标注在某个字段上的一致性得分"""
#     values = []
#     for a in anns:
#         v = a
#         for k in field:
#             v = v.get(k)
#         values.append(v)
#     if not values:
#         return 0.0
#     return Counter(values).most_common(1)[0][1] / len(values)


# def initial_confidence(anns: List[Dict]) -> float:
#     """
#     基于生成标注之间的一致性计算初始置信度分数
#     不涉及 Reviewer，仅基于 self-consistency
#     """
#     # if len(anns) < 2:
#     #     return 0.0
    
#     s_material = self_consistency(anns, ("material",))
#     s_trav = self_consistency(anns, ("visual_physical_hints", "traversability_hint"))
#     s_smoothness = self_consistency(anns, ("visual_attributes", "smoothness"))
#     s_moisture = self_consistency(anns, ("visual_attributes", "moisture"))
    
#     # 综合多个字段的一致性
#     return 0.4 * s_material + 0.3 * s_trav + 0.2 * s_smoothness + 0.1 * s_moisture


# def final_confidence(anns: List[Dict], reviewer_votes: List[bool]) -> float:
#     """
#     综合 self-consistency 和 reviewer 投票计算最终置信度
#     """
#     init_conf = initial_confidence(anns)
#     s_reviewer = sum(reviewer_votes) / len(reviewer_votes) if reviewer_votes else 0.0
#     return 0.7 * init_conf + 0.3 * s_reviewer


# def select_best_annotation(anns: List[Dict], reviewer_votes: Optional[List[bool]] = None) -> Tuple[Dict, float]:
#     """
#     基于置信度和 Reviewer 投票选择最佳标注
    
#     如果有 Reviewer 投票，优先选择被投赞成票且一致性高的标注
#     否则选择生成最多次的标注（majority voting）
#     """
#     if reviewer_votes:
#         # 有 Reviewer 复核，优先选择被投票通过的标注
#         best_idx = None
#         best_score = -1
        
#         for i, (annotation, vote) in enumerate(zip(anns, reviewer_votes)):
#             # Reviewer 通过且一致性高的标注优先
#             if vote:
#                 # 计算该标注与其他标注的平均相似度
#                 consistency_score = initial_confidence(anns)
#                 if consistency_score > best_score:
#                     best_score = consistency_score
#                     best_idx = i
        
#         # 如果没有被通过的标注，选择一致性最高的
#         if best_idx is None:
#             s_material = Counter([a["material"] for a in anns]).most_common(1)[0][1] / len(anns)
#             best_idx = 0
#             for i, ann in enumerate(anns):
#                 if ann["material"] == Counter([a["material"] for a in anns]).most_common(1)[0][0]:
#                     best_idx = i
#                     break
#     else:
#         # 没有 Reviewer 复核，选择生成最频繁的标注（material）
#         material_counter = Counter([a["material"] for a in anns])
#         most_common_material, _ = material_counter.most_common(1)[0]
        
#         best_idx = 0
#         for i, ann in enumerate(anns):
#             if ann["material"] == most_common_material:
#                 best_idx = i
#                 break
    
#     best_annotation = anns[best_idx]
#     confidence = final_confidence(anns, reviewer_votes if reviewer_votes else [])
#     return best_annotation, confidence


# ============================================================
# 主流程
# ============================================================

# ============================================================
# 主流程
# ============================================================

def run_pipeline(dataset_path: str, output_dir: Optional[str] = None):
    """
    运行标注流程
    
    Args:
        dataset_path: 数据集根目录路径
        output_dir: 输出目录路径（可选，默认为 ./Annotation/outputs）
    """
    # 设置数据集路径
    dataset_root = Path(dataset_path)
    if not dataset_root.exists():
        print(f"Error: Dataset path does not exist: {dataset_root}")
        sys.exit(1)
    
    # 获取数据集名称（最后一级目录名）
    dataset_name = dataset_root.name
    print(f"Processing dataset: {dataset_name}")
    print(f"Dataset path: {dataset_root}")
    
    # 设置输出目录
    if output_dir is None:
        output_base = Path("./Annotation/outputs")
    else:
        output_base = Path(output_dir)
    
    output_base.mkdir(parents=True, exist_ok=True)
    
    # 在 run_pipeline 函数中将所有预处理做好

    # client = OpenAI(api_key="YOUR_API_KEY")
    client_gpt = OpenAI(
        api_key="api_key",
        base_url="api_url"
    )
    client_qwen = OpenAI(
        api_key="api_key",
        base_url="api_url",
    )
    
    # 初始化全局 API 统计对象
    global_api_stats = APIStats()
    
    # completion = client.chat.completions.create(
    #     model="qwen-plus",
    #     messages=[{'role': 'user', 'content': '你是谁？'}]
    # )
    # print(completion.choices[0].message.content)

    # 遍历文件夹下所有的 local_image_final 目录
    for local_dir in dataset_root.rglob("local_image_final"):
        # 读取该目录中的标注JSON文件
        label_path = local_dir / "local_label.json"
        if not label_path.exists():
            print(f"Warning: {label_path} does not exist, skipping.")
            continue
        
        with open(label_path, "r") as f:
            items = json.load(f)["items"]
        
        results = []
        # 当前目录的 API 统计
        dir_api_stats = APIStats()

        for item in tqdm(items, desc=str(local_dir)):
            print(item)
            name = item["name"]
            # label = item["label"]
            tl, br = item["top_left"], item["bottom_right"]

            local_img = local_dir / name
            global_img = local_dir.parents[0] / "global_image" / f"{name[:10]}.png"
            boxed_global_img = draw_bbox(global_img, tl, br)

            # 输出local image 和 boxed global image 的路径方便调试
            # print(f"Local image path: {local_img}")
            # print(f"Global image path: {global_img}")

            # Resize 全景图到固定宽度
            h, w = boxed_global_img.shape[:2]
            scale = GLOBAL_IMAGE_WIDTH / w
            new_h = int(h * scale)
            boxed_global_img = cv2.resize(boxed_global_img, (GLOBAL_IMAGE_WIDTH, new_h))
            
            # boxed = output_base / f"boxed_{name}"
            # cv2.imwrite(str(boxed), boxed_global_img)
            # print(f"Saved resized boxed image to {boxed}")

            # existing_annotations截取item中已有的真值标注，包括category, material, function, traversability
            existing_annotations = {
                "category": item.get("category"),
                "material": item.get("material"),
                "function": item.get("function"),
                "traversability": item.get("traversability"),
            }

            # print(f"Processing {name} with existing annotations: {existing_annotations}")

            # 步骤1: 先用 Qwen 生成一次标注
            qwen_annotation = call_generator(
                client_qwen, 
                existing_annotations, 
                boxed_global_img, 
                local_img,
                annotation_model=GENERATOR_MODEL,
                api_stats=dir_api_stats
            )
            print(f"  [Qwen] Generator output: {qwen_annotation}")
            
            # 步骤2: 检查标注的有效性和置信度
            is_valid = check_annotation(qwen_annotation, existing_annotations) if qwen_annotation else False
            qwen_confidence = get_annotation_confidence(qwen_annotation) if qwen_annotation else 0.0
            
            print(f"  [Qwen] Valid: {is_valid}, Confidence: {qwen_confidence:.4f}")
            
            # 步骤3: 如果标注无效或置信度较低，则用 GPT 重新标注
            final_annotation = qwen_annotation
            model_used = "qwen"
            
            if not is_valid or qwen_confidence < CONF_WEAK:
                print(f"  [Qwen 不符合要求] 调用 GPT 进行重新标注...")
                gpt_annotation = call_generator(
                    client_gpt,
                    existing_annotations,
                    boxed_global_img,
                    local_img,
                    annotation_model=REVIEWER_MODEL,
                    api_stats=dir_api_stats
                )
                print(f"  [GPT] Generator output: {gpt_annotation}")
                
                if gpt_annotation:
                    is_valid_gpt = check_annotation(gpt_annotation, existing_annotations)
                    gpt_confidence = get_annotation_confidence(gpt_annotation)
                    print(f"  [GPT] Valid: {is_valid_gpt}, Confidence: {gpt_confidence:.4f}")
                    
                    # 选择更好的标注
                    if is_valid_gpt and gpt_confidence > qwen_confidence:
                        final_annotation = gpt_annotation
                        final_confidence = gpt_confidence
                        model_used = "gpt (retry)"
                    else:
                        final_annotation = qwen_annotation
                        final_confidence = qwen_confidence
                        model_used = "qwen"
                else:
                    final_annotation = qwen_annotation
                    final_confidence = qwen_confidence
                    model_used = "qwen"
            else:
                final_confidence = qwen_confidence
            
            if final_annotation:
                results.append({
                    "name": name,
                    "final_annotation": final_annotation,
                    "confidence": final_confidence,
                    "model_used": model_used,
                })

        # 保存标注结果
        out = local_dir / f"annotations.json"
        json.dump(results, open(out, "w"), indent=2, ensure_ascii=False)
        print(f"Saved annotations to {out}")
        
        # 保存该目录的 API 统计信息
        stats_out = output_base / f"{dataset_name}_api_stats.json"
        json.dump(dir_api_stats.to_dict(), open(stats_out, "w"), indent=2, ensure_ascii=False)
        print(f"Saved API stats to {stats_out}")
        
        # 累加到全局统计
        global_api_stats.generator_calls += dir_api_stats.generator_calls
        global_api_stats.generator_success += dir_api_stats.generator_success
        global_api_stats.generator_failed += dir_api_stats.generator_failed
        global_api_stats.reviewer_calls += dir_api_stats.reviewer_calls
        global_api_stats.reviewer_success += dir_api_stats.reviewer_success
        global_api_stats.reviewer_failed += dir_api_stats.reviewer_failed
    
    # 保存全局 API 统计信息
    global_stats_out = output_base / f"{dataset_name}_global_api_stats.json"
    json.dump(global_api_stats.to_dict(), open(global_stats_out, "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved global API stats to {global_stats_out}")
    print(f"Overall Statistics: {global_api_stats.to_dict()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Terrain annotation pipeline with Qwen and GPT models"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the dataset root directory"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="/8TBHDD3/tht/Annotation/outputs",
        help="Output directory for results (optional, default: ./Annotation/outputs)"
    )
    
    args = parser.parse_args()
    
    try:
        run_pipeline(args.dataset_path, args.output)
        print("\n✓ Pipeline completed successfully!")
    except Exception as e:
        print(f"\n✗ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
