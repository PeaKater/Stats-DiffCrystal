#!/bin/bash

# 设置CUDA可见设备（使用三张4090，根据你的环境可调整）
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONUNBUFFERED=1

# master 地址与端口（DDP需要，避免端口冲突可改）
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29512

# 获取当前脚本的PID
echo "当前脚本PID: $$"
echo "父进程PID (PPID): $PPID"

# 使用nohup后台运行（train.py 内部自动多卡并行）
nohup python -u train.py \
    --config_path ./conf/ssddpm.yaml \
    --output_path ./output_mp_7 \
    --expname Stats-DiffCrystal_multi_gpu \
    --early_stop 1200 \
    > training.log 2>&1 &

# 获取训练进程的PID
TRAIN_PID=$!
echo "训练进程PID: $TRAIN_PID"
echo "训练进程PID: $TRAIN_PID" > train_pid.txt

echo "训练已在后台启动"
echo "日志文件: training.log"
echo "进程ID已保存到: train_pid.txt"
echo ""
echo "要停止训练，请运行以下命令之一:"
echo "  kill $TRAIN_PID                    # 停止训练进程"
echo "  kill -9 $TRAIN_PID               # 强制停止训练进程"
echo "  pkill -f 'python train.py'       # 停止所有train.py进程"
echo "  pkill -P $$                      # 停止当前脚本的所有子进程"
echo ""
echo "查看训练日志: tail -f training.log"