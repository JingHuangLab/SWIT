#!/bin/bash
#SBATCH -J swit_rl
#SBATCH -N 1
#SBATCH -c 6
#SBATCH -p A40
#SBATCH --gres=gpu:1
#SBATCH -o /home/zhangky/rein_joblog/pred_%j.log
#SBATCH -e /home/zhangky/rein_joblog/pred_%j.err
 
module load mathlib/cuda/10.1.168_418.67
source /home/zhangky/miniconda3/bin/activate /home/zhangky/miniconda3/envs/molpal_w_rein

cd /home/zhangky/tool/swit/
python Reinvent2.0/input.py ./$1/RL_practice/test1/RL_config.json
