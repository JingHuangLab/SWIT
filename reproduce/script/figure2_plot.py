import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.pyplot import MultipleLocator
import os

### Change the folder to the path where the script is located
current_work_dir = os.getcwd()
os.chdir(current_work_dir)
print(f"current_work_dir:{current_work_dir}")

### Set drawing parameters and target name
mpl.rcParams['hatch.linewidth'] =1.5
current_palette = sns.color_palette("Accent")
hatch_plot = '\\'#['-', '', '\\', '-', '+', 'x', 'o', 'O', '.', '*','\\']
width=0.155
linewidth=3.5
fig=plt.figure(figsize=(42,42))
grid = plt.GridSpec(19,4,wspace=1,hspace=0.)
x=[0.1, 1.1, 2.1, 3.1,4.1,5.1,6.1,7.1]
rcpt_order=["AmpC",'D$_{4}$',"PARP1","JAK2","EGFR","PKM2","ALDH1","MAPK1"]

### Read data
read_df = pd.read_csv("../data/fig2_pcc_rmse.csv")
### Prediction results of model 1 on C1, V1
model1_c1_pcc=read_df[(read_df["model"]=="model1") & (read_df["dataset"]=="C1") ]["pcc"].tolist()
model1_c1_rmse=read_df[(read_df["model"]=="model1") & (read_df["dataset"]=="C1") ]["rmse"].tolist()
model1_v1_pcc=read_df[(read_df["model"]=="model1") & (read_df["dataset"]=="V1") ]["pcc"].tolist()
model1_v1_rmse=read_df[(read_df["model"]=="model1") & (read_df["dataset"]=="V1") ]["rmse"].tolist()

### Prediction results of model 2 on C1, V1 and V2
model2_c1_pcc=read_df[(read_df["model"]=="model2") & (read_df["dataset"]=="C1") ]["pcc"].tolist()
model2_c1_rmse=read_df[(read_df["model"]=="model2") & (read_df["dataset"]=="C1") ]["rmse"].tolist()
model2_v1_pcc=read_df[(read_df["model"]=="model2") & (read_df["dataset"]=="V1") ]["pcc"].tolist()
model2_v1_rmse=read_df[(read_df["model"]=="model2") & (read_df["dataset"]=="V1") ]["rmse"].tolist()
model2_v2_pcc=read_df[(read_df["model"]=="model2") & (read_df["dataset"]=="V2") ]["pcc"].tolist()
model2_v2_rmse=read_df[(read_df["model"]=="model2") & (read_df["dataset"]=="V2") ]["rmse"].tolist()

### Plot
ax1=plt.subplot(grid[0:4,0:4])
### pcc of top 4 targets
### model1 
ax1.bar(x[0:4],model1_c1_pcc[0:4],width=width,fc=current_palette[0], edgecolor='k',linewidth=linewidth,hatch='')#,yerr=model1_c1_pcc_err[0:4],tick_label=rcpt_order,fc=current_palette[9],hatch=hatch_par[]
for idx,i in enumerate(x[0:4]):
    ax1.text(i-0.08,model1_c1_pcc[idx]+0.05,str(round(model1_c1_pcc[idx],2)),rotation=60,fontsize=50)
for i in range(len(x)):
    x[i]=x[i]+width
ax1.bar(x[0:4],model1_v1_pcc[0:4],width=width,fc=current_palette[1], edgecolor='k',linewidth=linewidth,hatch='')#,tick_label=rcpt_order,fc=current_palette[0],yerr=model1_v1_pcc_err[0:4]
for idx,i in enumerate(x[0:4]):
    ax1.text(i-0.08,model1_v1_pcc[idx]+0.05,str(round(model1_v1_pcc[idx],2)),rotation=60,fontsize=50)
    
### model2
for i in range(len(x)):
    x[i]=x[i]+width+0.06
ax1.bar(x[0:4],model2_c1_pcc[0:4],width=width,fc=current_palette[0], edgecolor='k',linewidth=linewidth,hatch=hatch_plot,tick_label=rcpt_order[0:4])    
for idx,i in enumerate(x[0:4]):
    ax1.text(i-0.08,model2_c1_pcc[idx]+0.05,str(round(model2_c1_pcc[idx],2)),rotation=60,fontsize=50)
for i in range(len(x)):
    x[i]=x[i]+width
ax1.bar(x[0:4],model2_v1_pcc[0:4],width=width,fc=current_palette[1], edgecolor='k',linewidth=linewidth,zorder=1,hatch=hatch_plot)#,yerr=model2_v2_pcc_err[0:4]
for idx,i in enumerate(x[0:4]):
    ax1.text(i-0.08,model2_v1_pcc[idx]+0.05,str(round(model2_v1_pcc[idx],2)),rotation=60,fontsize=50)
for i in range(len(x)):
    x[i]=x[i]+width
ax1.bar(x[0:4],model2_v2_pcc[0:4],width=width,fc=current_palette[2],linewidth=linewidth, edgecolor='k',hatch=hatch_plot)#,tick_label=rcpt_order,fc=current_palette[0],yerr=model2_v2_pcc_err[0:4]
for idx,i in enumerate(x[0:4]):
    ax1.text(i-0.08,model2_v2_pcc[idx]+0.05,str(round(model2_v2_pcc[idx],2)),rotation=60,fontsize=50)
ax1.set_ylabel("PCC",size=70)#Pearson Correlation Coefficient\n(PCC)
ax1.text(-0.55,1.05,"a",size=75,fontweight='bold')
ax1.tick_params(labelsize=60)
ax1.set_yticks([0,0.5,1])
sns.despine()

### pcc of last 4 targets
ax1=plt.subplot(grid[5:9,0:4])
### C1
x=[0.1, 1.1, 2.1, 3.1,4.1,5.1,6.1,7.1]
ax1.bar(x[4:8],model1_c1_pcc[4:8],width=width,fc=current_palette[0], edgecolor='k',linewidth=linewidth,hatch='')#,,yerr=model1_c1_pcc_err[4:8]tick_label=rcpt_order,fc=current_palette[9],hatch=hatch_par[]
for idx,i in enumerate(x[4:8]):
    ax1.text(i-0.08,model1_c1_pcc[idx+4]+0.05,str(round(model1_c1_pcc[idx+4],2)),rotation=60,fontsize=50)
for i in range(len(x)):
    x[i]=x[i]+width
ax1.bar(x[4:8],model1_v1_pcc[4:8],width=width,fc=current_palette[1],linewidth=linewidth, edgecolor='k',hatch='')#,tick_label=rcpt_order,fc=current_palette[0],yerr=model1_v1_pcc_err[4:8]
for idx,i in enumerate(x[4:8]):
    ax1.text(i-0.08,model1_v1_pcc[idx+4]+0.05,str(round(model1_v1_pcc[idx+4],2)),rotation=60,fontsize=50)


### V1
for i in range(len(x)):
    x[i]=x[i]+width+0.06
ax1.bar(x[4:8],model2_c1_pcc[4:8],width=width,fc=current_palette[0],linewidth=linewidth, edgecolor='k',hatch=hatch_plot,tick_label=rcpt_order[4:8])
for idx,i in enumerate(x[4:8]):
    ax1.text(i-0.08,model2_c1_pcc[idx+4]+0.05,str(round(model2_c1_pcc[idx+4],2)),rotation=60,fontsize=50)
for i in range(len(x)):
    x[i]=x[i]+width
ax1.bar(x[4:8],model2_v1_pcc[4:8],width=width,fc=current_palette[1],linewidth=linewidth, edgecolor='k',zorder=1,hatch=hatch_plot)#,yerr=model2_v2_pcc_err[4:8]
for idx,i in enumerate(x[4:8]):
    ax1.text(i-0.08,model2_v1_pcc[idx+4]+0.05,str(round(model2_v1_pcc[idx+4],2)),rotation=60,fontsize=50)
for i in range(len(x)):
    x[i]=x[i]+width
ax1.bar(x[4:8],model2_v2_pcc[4:8],width=width,fc=current_palette[2],linewidth=linewidth, edgecolor='k',hatch=hatch_plot)#,tick_label=rcpt_order,fc=current_palette[0],yerr=model2_v2_pcc_err[4:8]
for idx,i in enumerate(x[4:8]):
    ax1.text(i-0.08,model2_v2_pcc[idx+4]+0.05,str(round(model2_v2_pcc[idx+4],2)),rotation=60,fontsize=50)
ax1.set_ylabel("PCC",size=70)#Pearson Correlation Coefficient\n(PCC)
ax1.tick_params(labelsize=60)
ax1.set_ylim(0,1)
ax1.set_yticks([0,0.5,1])
sns.despine()

### rmse of top 4 targets
ax2 = plt.subplot(grid[10:14,0:4])
### rmse of top 4 targets
### C1
x=[0.1, 1.1, 2.1, 3.1,4.1,5.1,6.1,7.1]
ax2.bar(x[0:4],model1_c1_rmse[0:4],width=width,fc=current_palette[0], edgecolor='k',linewidth=linewidth,hatch='')#,tick_label=rcpt_order,fc=current_palette[9],hatch=hatch_par[],yerr=model1_c1_rmse_err[0:4]
for idx,i in enumerate(x[0:4]):
    ax2.text(i-0.08,model1_c1_rmse[idx]+0.15,str(round(model1_c1_rmse[idx],2)),rotation=60,fontsize=50)
for i in range(len(x)):
    x[i]=x[i]+width
ax2.bar(x[0:4],model1_v1_rmse[0:4],width=width,fc=current_palette[1],linewidth=linewidth, edgecolor='k',hatch='')#,tick_label=rcpt_order,fc=current_palette[0],yerr=model1_v1_rmse_err[0:4]
for idx,i in enumerate(x[0:4]):
    ax2.text(i-0.08,model1_v1_rmse[idx]+0.15,str(round(model1_v1_rmse[idx],2)),rotation=60,fontsize=50)

### V1
for i in range(len(x)):
    x[i]=x[i]+width+0.06
ax2.bar(x[0:4],model2_c1_rmse[0:4],width=width,fc=current_palette[0], edgecolor='k',linewidth=linewidth,hatch=hatch_plot,tick_label=rcpt_order[0:4])
for idx,i in enumerate(x[0:4]):
    ax2.text(i-0.08,model2_c1_rmse[idx]+0.15,str(round(model2_c1_rmse[idx],2)),rotation=60,fontsize=50)
for i in range(len(x)):
    x[i]=x[i]+width
ax2.bar(x[0:4],model2_v1_rmse[0:4],width=width,fc=current_palette[1],linewidth=linewidth, edgecolor='k',zorder=1,hatch=hatch_plot)#,yerr=model2_v2_rmse_err[0:4]
for idx,i in enumerate(x[0:4]):
    ax2.text(i-0.08,model2_v1_rmse[idx]+0.15,str(round(model2_v1_rmse[idx],2)),rotation=60,fontsize=50)
for i in range(len(x)):
    x[i]=x[i]+width
ax2.bar(x[0:4],model2_v2_rmse[0:4],width=width,fc=current_palette[2],linewidth=linewidth, edgecolor='k',hatch=hatch_plot)#,tick_label=rcpt_order,fc=current_palette[0],yerr=model2_v2_rmse_err[0:4]
for idx,i in enumerate(x[0:4]):
    ax2.text(i-0.08,model2_v2_rmse[idx]+0.15,str(round(model2_v2_rmse[idx],2)),rotation=60,fontsize=50)
ax2.set_ylabel("RMSE",size=70)# Root Mean Square Error\n(RMSE)
ax2.tick_params(labelsize=60)
ax2.set_ylim(0,2.3)
ax2.text(-0.6,2.2,"b",size=75,fontweight='bold')
ax2.yaxis.set_major_locator(MultipleLocator(1))
ax2.yaxis.set_major_formatter('{x:.1f}')
sns.despine()

### rmse of last 4 targets
ax2=plt.subplot(grid[15:19,0:4])
### C1
ax2.bar(x[4:8],model1_c1_rmse[4:8],width=width,fc=current_palette[0], edgecolor='k',linewidth=linewidth,hatch='')#,tick_label=rcpt_order,,yerr=model1_c1_rmse_err[4:8]fc=current_palette[9],hatch=hatch_par[]
for idx,i in enumerate(x[4:8]):
    ax2.text(i-0.08,model1_c1_rmse[idx+4]+0.15,str(round(model1_c1_rmse[idx+4],2)),rotation=60,fontsize=50)
for i in range(len(x)):
    x[i]=x[i]+width
ax2.bar(x[4:8],model1_v1_rmse[4:8],width=width,fc=current_palette[1],linewidth=linewidth, edgecolor='k',hatch='')#,tick_label=rcpt_order,fc=current_palette[0],yerr=model1_v1_rmse_err[4:8]
for idx,i in enumerate(x[4:8]):
    ax2.text(i-0.08,model1_v1_rmse[idx+4]+0.15,str(round(model1_v1_rmse[idx+4],2)),rotation=60,fontsize=50)

### V1

for i in range(len(x)):
    x[i]=x[i]+width+0.06
ax2.bar(x[4:8],model2_c1_rmse[4:8],width=width,fc=current_palette[0], edgecolor='k',linewidth=linewidth,hatch=hatch_plot,tick_label=rcpt_order[4:8])
for idx,i in enumerate(x[4:8]):
    ax2.text(i-0.08,model2_c1_rmse[idx+4]+0.15,str(round(model2_c1_rmse[idx+4],2)),rotation=60,fontsize=50)
for i in range(len(x)):
    x[i]=x[i]+width
ax2.bar(x[4:8],model2_v1_rmse[4:8],width=width,fc=current_palette[1], edgecolor='k',zorder=1,linewidth=linewidth,hatch=hatch_plot)
for idx,i in enumerate(x[4:8]):
    ax2.text(i-0.08,model2_v1_rmse[idx+4]+0.15,str(round(model2_v1_rmse[idx+4],2)),rotation=60,fontsize=50)
for i in range(len(x)):
    x[i]=x[i]+width
ax2.bar(x[4:8],model2_v2_rmse[4:8],width=width,fc=current_palette[2],linewidth=linewidth, edgecolor='k',hatch=hatch_plot)#,tick_label=rcpt_order,fc=current_palette[0],yerr=model2_v2_rmse_err[4:8]
for idx,i in enumerate(x[4:8]):
    ax2.text(i-0.08,model2_v2_rmse[idx+4]+0.15,str(round(model2_v2_rmse[idx+4],2)),rotation=60,fontsize=50)
ax2.set_ylabel("RMSE",size=70)# Root Mean Square Error\n(RMSE)
ax2.tick_params(labelsize=60)
ax2.set_ylim(0,2.3)
ax2.yaxis.set_major_locator(MultipleLocator(1))
ax2.yaxis.set_major_formatter('{x:.1f}')
sns.despine()
legend_elements = [Patch(facecolor=current_palette[0], edgecolor='k',linewidth=linewidth,
                            label="C1"),
                    Patch(facecolor=current_palette[1], edgecolor='k',linewidth=linewidth,
                            label="V1"),
                    Patch(facecolor=current_palette[2], edgecolor='k',linewidth=linewidth,
                            label="V2"),
                Patch(facecolor="white", edgecolor='k',hatch='',linewidth=linewidth,
                        label="model1"),#model only using \nthe compounds of \nChemBridge for \ntraining
                Patch(facecolor="white", edgecolor='k',hatch=hatch_plot,linewidth=linewidth,
                        label="model2")]#model using the \ncombined dataset\n for training

fig.legend(handles=legend_elements,loc="center",prop={'size':65},bbox_to_anchor=(0.5,0.93),ncol=5,frameon=False)
plt.savefig("../result_graphs/figure2_metric_plot.png")#,dpi=100
     

