#!/bin/bash
# 脚本名称：run_dinov2_exp.sh
# 功能：依次运行dinov2_final.py从exp1到exp15，指定GPU 3，前一个失败则终止

# 可灵活配置的参数（按需修改）
SCRIPT_NAME="dinov2_final.py"  # 目标Python脚本名
GPU_ID=3                       # 指定使用的GPU编号
START_EXP=1                    # 起始实验编号
END_EXP=4                      # 结束实验编号

# 循环执行每个实验
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
    
    # 构建当前实验名称和执行命令
    EXP_NAME="exp${exp_num}"
    CMD="python ${SCRIPT_NAME} --exp_name ${EXP_NAME} --gpu ${GPU_ID}"
    
    # 打印执行上下文（便于日志追溯）
    echo "========================================"
    echo "开始执行实验：${EXP_NAME}"
    echo "执行命令：${CMD}"
    echo "执行时间：$(date +'%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    
    # 执行核心命令
    ${CMD}
    
    # 检查命令执行结果（0=成功，非0=失败）
    if [ $? -ne 0 ]; then
        echo -e "\n❌ 实验${EXP_NAME}执行失败！"
        echo "❌ 终止后续所有实验，脚本退出"
        exit 1
    fi
    
    # 执行成功提示
    echo -e "\n✅ 实验${EXP_NAME}执行成功！\n"
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
# echo "开始执行实验：${EXP_NAME}"
# echo "执行命令：${CMD}"
# echo "当前时间：$(date +'%Y-%m-%d %H:%M:%S')"
# echo "========================================"

# # 执行 exp21 命令
# ${CMD}

# # 检查 exp21 执行结果
# if [ $? -ne 0 ]; then
#     echo -e "\n❌ 实验${EXP_NAME}执行失败！"
#     echo "❌ 终止后续所有实验，退出脚本"
#     exit 1
# fi

# echo -e "\n✅ 实验${EXP_NAME}执行成功！\n"

# 所有实验完成的最终提示
echo "🎉 所有实验（exp1到exp${END_EXP}，已跳过exp7）均执行完成！"
exit 0