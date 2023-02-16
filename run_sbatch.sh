#!/bin/bash
#SBATCH -J swit
#SBATCH -N 1
#SBATCH -c 6
#SBATCH -p your_gpu_card_name
#SBATCH --gres=gpu:1
#SBATCH -o /your/path/of/log/pred_%j.log
#SBATCH -e /your/path/of/log/pred_%j.err
 
module load mathlib/cuda/10.1.168_418.67
source /your/path/of/conda/bin/activate /your/path/of/envs/env_name

## train and test target-specific model
cd /your/path/of/swit/
#python train_tss_model.py data/ampc_trainset_demo.csv task_test --testing_dataset_path data/ampc_testset_demo.csv --ncpu 6 
## 1. Please change to your own ckpt path
## 2. Please also change the test set to the data path you want to use
## 3. Please change the task name used during training
#python test_tss_model.py examples/task_test/lightning_logs/version_5434400/checkpoints/epoch=49-step=799.ckpt data/ampc_testset_demo.csv --task_name task_test --ncpu 6

## reinforcement learning
#python gen_models/input.py ./examples/task_test/RL_practice/test1/RL_config.json
