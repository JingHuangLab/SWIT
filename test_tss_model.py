from mpn_models.dmpnn import MPNN
from mpn_models import utils
import os
import argparse
import torch
from mpn_models import mpnn
import ray
from tqdm import tqdm
import numpy as np
from joblib import load
import math
import scipy.stats as stats
from datetime import datetime
import matplotlib.pyplot as plt

current_time=datetime.now()
suffix=str(current_time.year)+"_"+str(current_time.month)+"_"+str(current_time.day)

parser = argparse.ArgumentParser(description='Testing a target-specific scoring model.')
parser.add_argument('model_path', type=str, 
                    help='a path to a trained scoring model in ckpt format. It could be absolute path or relative path. Example: ./task_name/lightning_logs/version_0/checkpoints/epoch=4-step=79.ckpt')
parser.add_argument('testing_dataset_path', type=str,
                    help='a path to a file in csv format. It could be absolute path or relative path. Example: ./data/ampc_round2_test1k.csv The first column should be SMILES and second column should be score.')
parser.add_argument('--task_name', type=str, default="data",required=False,
                    help='name of the output folder located under swit/. default : data')
parser.add_argument('--ncpu', type=int, default=1,required=False,
                    help='the number of cores to available to each worker / job / process / node. default : 1')
args = parser.parse_args()
if not os.path.exists("./"+args.task_name+"/preds"):
    cmd="mkdir -p ./"+args.task_name+"/preds"
    os.system(cmd)
current_work_dir = os.getcwd()
os.chdir(current_work_dir)
### load data
scores_csv       = args.testing_dataset_path
scores, failures = utils._read_scores(scores_csv)
xs, ys           = zip(*scores.items())
print(f"input data size:{len(xs)}")

### load ckpt
dict_pt = torch.load(args.model_path)["state_dict"]
new_dict = {}
for k in dict_pt.keys():
    ki = k.split("mpnn.")[1]
    new_dict[ki] = dict_pt[k]
    
### load model
tss_model = MPNN()
tss_model.model.load_state_dict(new_dict)
tss_model.model.eval() 

### Generate predictions for the inputs xs
_predict = ray.remote(num_cpus=args.ncpu, num_gpus=1)(mpnn.predict)  
use_gpu = ray.cluster_resources().get('GPU', 0) > 0
smis_batches = utils.batches(xs, 10000)#00
model = ray.put(tss_model.model)
scaler_path=args.model_path.split("checkpoints/epoch=")[0]+"std_scaler.bin"
scaler = ray.put(load(scaler_path))
batch_size=50
refs = [
    _predict.remote(
        model, smis, batch_size, args.ncpu,
        True, scaler, use_gpu, True
    ) for smis in smis_batches
]
preds_chunks = [
    ray.get(r) for r in tqdm(refs, desc='Prediction', leave=False)
]
pred_scores=[]
for idx in range(len(preds_chunks)):
    for pred_score in preds_chunks[idx][0]:
        pred_scores.append(pred_score[0])
print(f"output data size:{len(pred_scores)}")

### calculate RMSE,PCC between predicted score and target score
if len(pred_scores)==len(ys):
    se_lst = []
    writer=open("./"+args.task_name+"/preds/prediction"+suffix+".csv","w")
    writer.write("prediction,target\n")
    new_pred=[]
    new_ys=[]
    for idx,target_score in enumerate(ys):
        if target_score>=0:  ## filter that should not be the items
            continue
        se = (target_score-pred_scores[idx])**2
        se_lst.append(se)
        new_pred.append(pred_scores[idx])
        new_ys.append(target_score)
        writer.write(str(round(pred_scores[idx],1))+","+str(target_score)+"\n")
    writer.close()
    mse = np.average(se_lst)
    rmse = math.sqrt(mse)
    PCC,p =stats.pearsonr(new_pred,new_ys)
    print(f"length of score list:{len(se_lst)}")
    print(f"RMSE:{rmse:.2f}")
    print(f"PCC :{PCC:.2f}")
    ## plot
    fig=plt.figure(figsize=(15,15))
    x = np.linspace(-19,-1,19)
    y = x
    plt.scatter(pred_scores,ys,color='orange',s=6)
    plt.xlim(-20,0)
    plt.ylim(-20,0)
    plt.tick_params(labelsize=20)
    plt.plot(x, y,"b--")
    plt.xlabel("prediction",size=30)
    plt.ylabel("target",size=30)
    plt.savefig("./"+args.task_name+"/preds/pred_target_scatter"+suffix+".png")
else:
    print("Warning: the length of the prediciton and target are not the same.")


