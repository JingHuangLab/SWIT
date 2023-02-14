import os
import numpy as np
from numpy import *
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import MultipleLocator
from matplotlib.patches import Patch

current_work_dir = os.getcwd()
os.chdir(current_work_dir)
print(current_work_dir)
rcpt_name_lst = ["ampc","d4","parp1","jak2","egfr","pkm2","aldh1","mapk1"]#"cdk2",
cut_off_chems=["-11.3","-14.5","-13.6","-14.6","-14.1","-15.8","-13.9","-11.0"]#,"-13.6"
width=0.3
alpha=1
FIG = plt.figure(figsize=(80,80))
plt.subplots_adjust(wspace =0.2, hspace =0.4)
pos_lst=[(0,0),(0,2),(1,0),(1,2),(2,0),(2,2),(3,0),(3,2)]
current_palette = sns.color_palette("Paired")
legend_elements1 = [Patch(facecolor=current_palette[0], edgecolor='k',linewidth=1,
                        label="Generated molecules")]
legend_elements2 = [Patch(facecolor=current_palette[1], edgecolor='k',linewidth=1,
                        label="ChemBridge library")]
for rdx,rcpt_name in enumerate(rcpt_name_lst):
    print(rcpt_name.upper())
    cut_off_chem = cut_off_chems[rdx]
    
    ### prepare standard name for title of subplot
    if rcpt_name=="ampc":
        std_name = "AmpC"
    elif rcpt_name=="d4":
        std_name='D$_{4}$'
    else:
        std_name = rcpt_name.upper()
        
    ### read
    total_df=pd.read_csv("../data/figure3_score_ef_distribution.csv")
    temp_df = total_df[(total_df["receptor"]==std_name) & (total_df["data type"]=="score")]
    dock_score_cb=temp_df["hits number in ChemBridge"].to_list()
    target_score=temp_df["hits number in generated molecules"].to_list()
    ef_lst=total_df[(total_df["receptor"]==std_name) & (total_df["data type"]=="ef")]["other info"].to_list()
    ef_lst=list(map(float,ef_lst))
    x_lst_score=temp_df["threshold"].to_list()
    x_lst_ef=total_df[(total_df["receptor"]==std_name) & (total_df["data type"]=="ef")]["threshold"].to_list()

    ## #reverse
    x_lst_score.reverse()
    target_score.reverse()
    dock_score_cb.reverse()
    x_lst_ef.reverse()
    ef_lst.reverse()

    fig00=plt.subplot2grid((4,4),pos_lst[rdx])
    ## Make the x-axis spaced by 2
    x=list(range(len(x_lst_score)))
    x_lst_score_str=[]
    fig01_xticks=[]
    for idx,i in enumerate(x_lst_score):
        if idx%2==0:
            x_lst_score_str.append(str(int(i)))
            fig01_xticks.append(i)
        else:
            x_lst_score_str.append("")
            
    ## plot score distribution
    fig00.bar(x,target_score,width=width,label="Generated molecules",tick_label=x_lst_score_str,fc=current_palette[0],alpha=alpha, edgecolor='k',linewidth=1)
    for idx,i in enumerate(x):
        if x_lst_score[i]<=float(cut_off_chem):
            if target_score[idx]>3000:
                increase_m =target_score[idx]
                fig00.text(i-0.3,target_score[idx]+increase_m,str(target_score[idx]),fontsize=90)
            else:
                increase_m =target_score[idx]*0.5
                fig00.text(i-0.3,target_score[idx]+increase_m,str(target_score[idx]),fontsize=90)
    for i in range(len(x)):
        x[i]=x[i]+width
    fig00.bar(x,dock_score_cb,width=width,label="ChemBridge",fc=current_palette[1], edgecolor='k',linewidth=1)
    fig00.set_yscale('log')
    fig00.set_ylim(bottom=1,top=1000000)
    pos = (pos_lst[rdx][0],pos_lst[rdx][1]+1)
    fig01=plt.subplot2grid((4,4),pos)
    
    ## plot ef curve
    print("x_lst_ef:",x_lst_ef)
    print("ef_lst:",ef_lst)
    fig01.scatter(x_lst_ef,ef_lst,marker='o',c='none',edgecolors=current_palette[9],s=1200,linewidths=12,zorder=20)
    fig01.plot(x_lst_ef,ef_lst,color=current_palette[8],linestyle='dashed',linewidth=24,zorder=10,alpha=0.8)
    x_major_locator=MultipleLocator(2)
    my_x_ticks = np.arange(math.floor(x_lst_ef[-1]),math.ceil(x_lst_ef[0])+1)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_ticks_position('right')
    fig01.set_xlim(int(x_lst_score_str[0])+0.2,int(x_lst_score_str[-1])-0.9)
    fig01.set_xticks(ticks=fig01_xticks)
    fig01.set_yscale('log')
    fig01.set_ylim(bottom=1,top=10000)
    if rdx ==0:
        fig00.legend(handles=legend_elements1,bbox_to_anchor=(0.,1.35,1.,0.102),loc="upper left",prop={'size':120},ncol=1,mode="expand",borderaxespad=0.,frameon=False)
    if rdx ==1:
        fig00.legend(handles=legend_elements2,bbox_to_anchor=(0.,1.35,1.,0.102),loc="upper left",prop={'size':120},ncol=1,mode="expand",borderaxespad=0.,frameon=False)
    if rdx in [0,2,4,6]:
        fig00.tick_params(labelsize=105)
        fig01.tick_params(labelsize=105,direction="in",length=40,width=5,top=False,bottom=False,left=False,right=True)
        fig01.set_yticklabels([])
        fig01.spines['left'].set_visible(False)
        fig01.spines['top'].set_visible(False)
        fig00.spines['top'].set_visible(False)
        fig00.spines['right'].set_visible(False)
        
    else:
        fig00.tick_params(labelsize=105,direction="in",length=40,width=5,top=False,bottom=False,left=True,right=False)
        fig01.tick_params(labelsize=105)
        fig00.set_yticklabels([])
        fig01.spines['left'].set_visible(False)
        fig01.spines['top'].set_visible(False)
        fig00.spines['top'].set_visible(False)
        fig00.spines['right'].set_visible(False)
        
FIG.text(0.5,0.05,"Vina score threshold (kcal/mol)",size=135,ha="center")
FIG.text(0.05,0.5,"Number of hits",size=135,va="center",rotation='vertical')
FIG.text(0.95,0.5,"Enrichment factor",size=135,va="center",rotation=270)
rcpt_order=["AmpC",'D$_{4}$',"PARP1","JAK2","EGFR","PKM2","ALDH1","MAPK1"]
FIG.text(0.310,0.885,rcpt_order[0],size=135,ha="center")
FIG.text(0.715,0.885,rcpt_order[1],size=135,ha="center")
FIG.text(0.315,0.685,rcpt_order[2],size=135,ha="center")
FIG.text(0.715,0.685,rcpt_order[3],size=135,ha="center")
FIG.text(0.315,0.475,rcpt_order[4],size=135,ha="center")
FIG.text(0.715,0.475,rcpt_order[5],size=135,ha="center")
FIG.text(0.315,0.265,rcpt_order[6],size=135,ha="center")
FIG.text(0.715,0.265,rcpt_order[7],size=135,ha="center")
plt.savefig("../result_graphs/figure3_vina_score_ef.png")


