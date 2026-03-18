#!/bin/bash

SCRIPT_NAME="train_clip_final_benchmark3.py"

# 定义实验和对应要使用的 GPU 编号
# 格式: "实验名:GPU编号"
TASKS=(
    # "exp1:0"
    # "exp2:1"
    # "exp3:1"
    "exp4:1"
    "exp5:1"
)

if [ ! -f "$SCRIPT_NAME" ]; then
    echo "❌ 错误: 找不到文件 '$SCRIPT_NAME'！"
    exit 1
fi

echo "🚀 开始多卡并发运行实验..."
echo "======================================================="

for TASK in "${TASKS[@]}"; do
    # 解析实验名和 GPU 编号
    EXP="${TASK%%:*}"
    GPU="${TASK##*:}"
    
    echo "▶️ 正在后台启动实验: $EXP | 分配 GPU: $GPU | 日志将保存至 ${EXP}.log"
    
    # 使用 nohup 在后台运行，并将输出重定向到对应的日志文件中
    # 末尾的 & 表示放入后台执行
    nohup python $SCRIPT_NAME --exp_name $EXP --gpu $GPU > "${EXP}_run.log" 2>&1 &
done

echo "======================================================="
echo "✅ 所有任务已提交至后台运行！"
echo "你可以使用 'tail -f exp1_run.log' 等命令来实时查看各个实验的进度。"