import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import math
import seaborn as sns

current_work_dir = os.getcwd()
os.chdir(current_work_dir)
plt.style.use('seaborn-white')
FIG = plt.figure(figsize=(60,45))
plt.subplots_adjust(wspace =0., hspace =0.2)
current_palette = sns.color_palette("tab10")
ax3 = FIG.add_subplot(311)

## when the no.columns =3
line1=Line2D([.23,0.33],[.15,0.15],linestyle='-',color=current_palette[0],lw=11)
line2=Line2D([.23,0.33],[.05,0.05],linestyle='-',color=current_palette[1],lw=11)
line3=Line2D([.43,0.53],[.15,0.15],linestyle='-',color=current_palette[2],lw=11)
line4=Line2D([.43,0.53],[.05,0.05],linestyle='-',color=current_palette[3],lw=11)
line5=Line2D([.65,0.75],[.15,0.15],linestyle='-',color=current_palette[4],lw=11)
line6=Line2D([.65,0.75],[.05,0.05],linestyle='-',color=current_palette[5],lw=11)
line_lst=[line1,line2,line3,line4,line5,line6]
for line in line_lst:
    ax3.add_line(line)
ax3.text(0.19, 0.28, "Steric Interaction", fontsize=70)
ax3.text(0.37, 0.28, "Hydrophobic Interaction", fontsize=70)
ax3.text(0.63, 0.28, "Hydrogen Bond", fontsize=70)# Interaction
ax3.text(0.01, 0.14, "ChemBridge library", fontsize=70)
ax3.text(0.00, 0.02, "Generated molecules", fontsize=70)
ax3.axis("off")

rcpt_name_lst = ["ampc","d4","parp1","jak2","egfr","pkm2","aldh1","mapk1"]
sub_lst =[5,6,7,8,9,10,11,12]
ax_0_lst=[0,0,0,0,1,1,1,1]
ax_1_lst=[0,1,2,3,0,1,2,3]
alpha_value=0.8
bin_size=1
hue_order=["steric Chembridge","steric Generated set","hydrophobic Chembridge","hydrophobic Generated set","hydrogen Chembridge","hydrogen Generated set"]
for rdx,rcpt_name in enumerate(rcpt_name_lst):
    chem_df = pd.read_csv("../data/score_split_"+rcpt_name+"_top1000.txt",sep="\t")
    round2_df = pd.read_csv("../data/score_split_"+rcpt_name+"_round2_top1000.txt",sep="\t")#lip5_2
    if rcpt_name =="ampc":
        std_name="AmpC"
    elif rcpt_name=="d4":
        std_name='D$_{4}$'
    else:
        std_name=rcpt_name.upper()
    if chem_df.shape[0]>1000:
        chem_df = chem_df.sort_values(by="affinity",axis=0).iloc[0:1000,]
    else:
        chem_df = chem_df.sort_values(by="affinity",axis=0)
    round2_df = round2_df.sort_values(by="affinity",axis=0).iloc[0:1000,]
    class_lst=["Generated set"]*round2_df.shape[0]
    round2_df.insert(round2_df.shape[1],"Data set",class_lst)
    class_lst=["Chembridge"]*chem_df.shape[0]
    chem_df.insert(chem_df.shape[1],"Data set",class_lst)
    average_score_chem=chem_df["affinity"].mean()
    average_score_round2=round2_df["affinity"].mean()
    
    merge_df =pd.concat([chem_df,round2_df])
    merge_df.index=list(range(merge_df.shape[0]))
    df=pd.DataFrame(columns=["x","hue"])
    
    for index,row in merge_df.iterrows():
        ste=row['steric_sum']
        dat=row["Data set"]
        hydrog=row['hydrogen_value']
        hydrop=row['hydrophobic_value']
        df.loc[df.shape[0]]=[ste,'steric '+dat]
        df.loc[df.shape[0]]=[hydrog,'hydrogen '+dat]
        df.loc[df.shape[0]]=[hydrop,'hydrophobic '+dat]
        
    print(sub_lst[rdx])
    ax=FIG.add_subplot(3,4,sub_lst[rdx])#(sub_lst[rdx])
    bin=np.arange(math.floor(min(df['x'])/bin_size)*bin_size,math.ceil(max(df['x'])/bin_size)*bin_size+bin_size,bin_size)
    sns.histplot(df,x='x',hue="hue",palette=current_palette[0:4]+current_palette[4:6],ax=ax,bins=bin,stat="probability",common_norm=False,kde=True,line_kws={'linewidth':11},hue_order=hue_order,edgecolor=None,fill=False,linewidth=0)#,palette='tab10'
    ax.set_xlim([-20,3])
    ax.set_ylim([0,1])
    ax.tick_params(labelsize=70)
    ax.set_title(std_name,size=80)
    ax.legend_.remove()
    ax.set(ylabel=None)
    ax.set(xlabel=None)
    if not rdx in [0,4]:
        ax.set_yticks([])
    if not rdx in [4,5,6,7,]:
        ax.set_xticks([])

FIG.text(0.5,0.06,"kcal/mol",size=90,ha="center")#0.5,0.34
FIG.text(0.08,0.32,"Probability",size=90,ha="center",rotation='vertical')#0.08,0.57
plt.savefig("../result_graphs/figure4_score_split.png")