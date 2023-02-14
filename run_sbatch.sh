#!/bin/bash
#SBATCH -J swit_train
#SBATCH -N 1
#SBATCH -c 6
#SBATCH -p A40
#SBATCH --gres=gpu:1
#SBATCH -o /home/zhangky/rein_joblog/pred_%j.log
#SBATCH -e /home/zhangky/rein_joblog/pred_%j.err
 
module load mathlib/cuda/10.1.168_418.67
source /home/zhangky/miniconda3/bin/activate /home/zhangky/miniconda3/envs/molpal

cd /home/zhangky/tool/swit/models
#python train_tss_model.py ../data/ampc_trainset.csv ../data/ampc_testset.csv ampc_model1 --ncpu 6 
python test_tss_model.py ../ampc_model1/lightning_logs/version_5433864/checkpoints/epoch=40-step=433697.ckpt ../data/ampc_gen1_test.csv --task_name ampc_model1 --ncpu 6
