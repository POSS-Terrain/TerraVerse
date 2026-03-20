#!/bin/bash
# 脚本名称：run_mobilenet_exp.sh
# 功能：依次运行MobileNet_final.py从exp1到exp15，指定GPU 0，前一个失败则终止，最后跑exp21

# 定义基础参数
SCRIPT_NAME="MobileNet_final.py"
GPU_ID=0
START_EXP=1
END_EXP=4

# 循环执行 exp1 到 exp15
for (( exp_num=START_EXP; exp_num<=END_EXP; exp_num++ ))
do
    # 修改：跳过exp7
    if [ ${exp_num} -eq 7 ]; then
        echo "========================================"
        echo "🔶 跳过实验exp${exp_num}，继续下一个实验"
        echo "========================================"
        echo ""
        continue  # 跳过本次循环，执行下一个exp_num
    fi

    # 构建当前要执行的命令
    EXP_NAME="exp${exp_num}"
    CMD="python ${SCRIPT_NAME} --exp_name ${EXP_NAME} --gpu ${GPU_ID}"
    
    # 打印当前执行信息（方便日志查看）
    echo "========================================"
    echo "开始执行：${CMD}"
    echo "当前时间：$(date +'%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    
    # 执行命令
    ${CMD}
    
    # 检查命令执行结果，非0表示失败
    if [ $? -ne 0 ]; then
        echo "❌ 执行失败：${CMD}"
        echo "❌ 实验${EXP_NAME}执行出错，终止后续所有实验"
        exit 1
    fi
    
    # 执行成功提示
    echo "✅ 执行成功：${CMD}"
    echo ""
done

# # ==============================================================================
# # 🚩 新增：单独执行 exp21
# # ==============================================================================
# echo "========================================"
# echo "🎉 阶段一完成 (exp1-exp15)，开始执行 exp21"
# echo "========================================"

# EXP_NAME="exp21"
# CMD="python ${SCRIPT_NAME} --exp_name ${EXP_NAME} --gpu ${GPU_ID}"

# echo "========================================"
# echo "开始执行：${CMD}"
# echo "当前时间：$(date +'%Y-%m-%d %H:%M:%S')"
# echo "========================================"

# # 执行 exp21 命令
# ${CMD}

# # 检查 exp21 执行结果
# if [ $? -ne 0 ]; then
#     echo "❌ 执行失败：${CMD}"
#     echo "❌ 实验${EXP_NAME}执行出错"
#     exit 1
# fi

# echo "✅ 执行成功：${CMD}"
# echo ""

# 所有实验完成提示
echo "🎉 所有实验（exp1到exp${END_EXP}）均执行完成！"
exit 0