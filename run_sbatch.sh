#!/bin/bash
#SBATCH -J swit_train
#SBATCH -N 1
#SBATCH -c 6
#SBATCH -p your_card_name
#SBATCH --gres=gpu:1
#SBATCH -o /your/path/of/log/pred_%j.log
#SBATCH -e /your/path/of/log/pred_%j.err
 
module load mathlib/cuda/10.1.168_418.67
source /your/path/of/conda/bin/activate /your/path/of/envs/env_name

cd /your/path/of/swit/models
#python train_tss_model.py ../data/ampc_trainset.csv ../data/ampc_testset.csv ampc_model1 --ncpu 6 
python test_tss_model.py ../task-name/ightning_logs/version_x/checkpoints/xxx.ckpt ../data/ampc_testset521.csv --task_name your_task_name --ncpu 6
