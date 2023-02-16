from mpn_models.dmpnn import MPNN
from mpn_models.chemprop import *
from mpn_models import utils
import os
import argparse
import sys
from joblib import dump
import numpy as np
import math
import scipy.stats as stats
from datetime import datetime
import matplotlib.pyplot as plt


current_time=datetime.now()
suffix=str(current_time.year)+"_"+str(current_time.month)+"_"+str(current_time.day)

parser = argparse.ArgumentParser(description='Train a target-specific scoring model.')
parser.add_argument('training_dataset_path', type=str, 
                    help='a path to a file in csv format. It could be absolute path or relative path. Example: ./data/ampc_round2_test1k.csv The first column should be SMILES and second column should be score.')
parser.add_argument('task_name', type=str, default=None,
                    help='the name of the output folder located under swit/')
parser.add_argument('--testing_dataset_path', type=str, default=None,
                    help='a path to a file in csv format. It could be absolute path or relative path. Example: ./data/ampc_round2_test1k.csv The first column should be SMILES and second column should be score. default: None')
parser.add_argument('--ncpu', type=int, default=1,
                    help='number of cores to available to each worker/job/process/node. default: 1')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of iterations for model training. default: 50')
args = parser.parse_args()

if not os.path.exists("./"+args.task_name+"/preds"):
    cmd="mkdir -p ./"+args.task_name+"/preds"
    os.system(cmd)
current_work_dir = os.getcwd()
os.chdir(current_work_dir)
scores_csv       = args.training_dataset_path
scores, failures = utils._read_scores(scores_csv)
xs, ys           = zip(*scores.items())
print(f"input data size:{len(xs)}")

my_model= MPNN(ncpu=args.ncpu,epochs=args.epochs)
my_model.train(xs, ys,args.task_name)
#print(my_model)
print("Trained model saved in /swit/"+args.task_name+"...")
folder_name=sorted(os.listdir("./"+args.task_name+"/lightning_logs"))[-1]
dump(my_model.scaler,f'./'+args.task_name+'/lightning_logs/'+folder_name+'/std_scaler.bin',compress=True)
print("The scaler is already stored in {}".format('./'+args.task_name+'/lightning_logs/'+folder_name+'/std_scaler.bin'))

### prediction
if not args.testing_dataset_path is None:
    ### load data
    scores_csv       = args.testing_dataset_path
    scores, failures = utils._read_scores(scores_csv)
    xs, ys           = zip(*scores.items())
    print(f"input data size:{len(xs)}")
    preds_chunks=my_model.predict(xs)
    pred_scores=[]
    for idx in range(len(preds_chunks)):
        for pred_score in preds_chunks[idx]:
                pred_scores.append(pred_score[0])
    print(f"output data size:{len(pred_scores)}")

    ### calculate RMSE,PCC between predicted score and target score
    if len(pred_scores)==len(ys):
        se_lst = []
        new_pred=[]
        new_ys=[]
        writer=open("./"+args.task_name+"/preds/prediction"+suffix+".csv","w")
        writer.write("prediction,target\n")
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
