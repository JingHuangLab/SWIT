import math
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

### Change the folder to the path where the script is located
current_work_dir = os.getcwd()
os.chdir(current_work_dir)
print(f"current_work_dir:{current_work_dir}")

### Set drawing parameters and target name
rcpt_names=["pkm2","aldh1"]
current_palette = sns.color_palette("tab10")
fig=plt.figure(figsize=(50,20))
window_len = 53
polyorder = 3
mod = "mirror"
legend_lst=[]
for rdx,rcpt_name in enumerate(rcpt_names):
    
    ### Read data
    pkm2_time_df=pd.read_csv("../data/time_record/time_record_"+rcpt_name+"_100k_steps.csv",header=None)
    pkm2_time_df.columns=["status","current_time"]
    start=pkm2_time_df.loc[0]["current_time"]
    pkm2_time_df["total time"]=(pkm2_time_df["current_time"]-start)/3600
    pkm2_time_df=pkm2_time_df.drop(0)
    
    hour_epoch_df=pd.read_csv("../data/time_record/hour_and_new_scaffolds_record_"+rcpt_name+".csv")
    std_name=rcpt_name.upper()
    x=hour_epoch_df["Nth hour"].to_list()
    x_of_6500=math.floor(pkm2_time_df[pkm2_time_df["status"]=="6500"]["total time"].values[0])
    print(pkm2_time_df[pkm2_time_df["status"]=="6500"]["total time"].values[0])
    
    ### Plot
    plt.ylabel('Number of novel potential hits',size=50)
    y=hour_epoch_df["number of new scaffolds"].to_list()
    plt.yticks([5000,10000,15000],labels=("5,000","10,000","15,000"))
    plt.ylim(0,15000)
    plt.xlim(0,331)

    ### Smooth curve
    y_smooth = savgol_filter(y,window_len,polyorder,mode=mod)
    new_df=pd.DataFrame({"x":x,"y_smooth":y_smooth})
    new_df.to_csv("../data/time_record/"+rcpt_name+"_x_y_smooth.csv",index=False)
    line,=plt.plot(x,y_smooth,'-',color=current_palette[rdx],linewidth=6,label=std_name)
    legend_lst.append(line)
    plt.xlabel("GPU hours",size=50)
    plt.axvline(x=x_of_6500,color="black",linewidth=6,linestyle="--")
    plt.tick_params(labelsize=40)
    plt.legend(handles=legend_lst,loc=1,prop={'size':40},frameon=False,ncol=2)#,bbox_to_anchor=(1.03,0.001)
    plt.annotate('', (15,8484), xytext=(0, -40),textcoords='offset points', arrowprops=dict(facecolor=current_palette[1], shrink=0,edgecolor=current_palette[1]),size=40)
    plt.annotate('', (32,12710), xytext=(40, 0),textcoords='offset points', arrowprops=dict(facecolor=current_palette[0], shrink=0,edgecolor=current_palette[0]),size=40)
    plt.annotate('', (100,6910), xytext=(0,40),textcoords='offset points', arrowprops=dict(facecolor=current_palette[0], shrink=0,edgecolor=current_palette[0]),size=40)
    plt.annotate('', (100,6958), xytext=(0,-40),textcoords='offset points', arrowprops=dict(facecolor=current_palette[1], shrink=0,edgecolor=current_palette[1]),size=40)
    
    plt.annotate('', (200,6515), xytext=(0,40),textcoords='offset points', arrowprops=dict(facecolor=current_palette[0], shrink=0,edgecolor=current_palette[0]),size=40)
    plt.annotate('', (200,5078), xytext=(0,-40),textcoords='offset points', arrowprops=dict(facecolor=current_palette[1], shrink=0,edgecolor=current_palette[1]),size=40)
    
    plt.annotate('', (300,6594), xytext=(0,40),textcoords='offset points', arrowprops=dict(facecolor=current_palette[0], shrink=0,edgecolor=current_palette[0]),size=40)
    plt.annotate('', (300,3574), xytext=(0,-40),textcoords='offset points', arrowprops=dict(facecolor=current_palette[1], shrink=0,edgecolor=current_palette[1]),size=40)
    print("mean:{};median:{};max:{};min:{}".format(round(np.mean(y),2),round(np.median(y),2),round(np.max(y),2),round(np.min(y),2)))
plt.savefig("../result_graphs/figure7_num_pertential_hits.png", bbox_inches='tight',pad_inches=0.5)#_smooth2





