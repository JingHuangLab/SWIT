import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean
from numpy.lib.function_base import median
import seaborn as sns
from matplotlib.patches import Patch
import os

### Change the folder to the path where the script is located
current_work_dir = os.getcwd()
os.chdir(current_work_dir)
print(f"current_work_dir:{current_work_dir}")

### Set drawing parameters and target name
rcpt_name_lst = ["ampc","d4","parp1","jak2","egfr","pkm2","aldh1","mapk1"]#
FIG = plt.figure(figsize=(60,42))
plt.subplots_adjust(wspace =0.05, hspace =0.15)
sub_lst =[241,242,243,244,245,246,247,248]
current_palette = sns.color_palette("Set2")
for rdx,rcpt_name in enumerate(rcpt_name_lst):
    main_name = rcpt_name+"_round2"
    print("________________________________")
    print(main_name.upper())
    if rcpt_name=="ampc":
        stdname = "AmpC"
    elif rcpt_name=="d4":
        stdname='D$_{4}$'
    else:
        stdname = rcpt_name.upper()

    ### Read data
    ### ChEMBL
    group_simi = []
    label_lst = []
    simi_eq1 =0
    reader = open("../data/merged_similarity/"+main_name+"_withchembl_merge_top100k.txt","r")
    for ldx,line in enumerate(reader):
        if "top_id" in line:
            continue
        if len(line.split("\t"))==1:
            continue
        simi = line.split("\t")[3].strip()
        group_simi.append(float(simi))
        label_lst.append("ChEMBL")
        if float(simi)==1.0:
            simi_eq1+=1
    reader.close()
    print(str(len(label_lst)))
    len2=0
    print("COMPARE WITH CHEMBL: max:{:.2f};min:{:.2f};average:{:.2f};median:{:.2f}".format(max(group_simi[len2:]),min(group_simi[len2:]),mean(group_simi[len2:]),median(group_simi[len2:])))
    print("Number of molecules with a similarity of 1 to ChEMBL:{}\n".format(simi_eq1))
    len2=len(group_simi)

    ### ChemBridge
    reader = open("../data/merged_similarity/"+main_name+"_withchembridge_merge_top100k.txt","r")
    simi_eq3=0
    for ldx,line in enumerate(reader):
        if "top_id" in line:
            continue
        if len(line.split("\t"))==1:
            continue
        simi = line.split("\t")[3].strip()
        group_simi.append(float(simi))
        label_lst.append("ChemBridge")
        if float(simi)==1.0:
            simi_eq3+=1
    reader.close()
    print(str(len(label_lst)))
    len3=len(group_simi)
    print("COMPARE WITH CHEM: max:{:.2f};min:{:.2f};average:{:.2f};median:{:.2f}".format(max(group_simi[len2:]),min(group_simi[len2:]),mean(group_simi[len2:]),median(group_simi[len2:])))
    print("Number of molecules with a similarity of 1 to ChemBridge:{}\n".format(simi_eq3))

    ### ZINC20
    simi_eq2=0
    reader = open("../data/merged_similarity/"+main_name+"_withzinc_merge_top100k.txt","r")
    for ldx,line in enumerate(reader):
        if "top_id" in line:
            continue
        simi = line.split("\t")[3]
        group_simi.append(float(simi))
        label_lst.append("ZINC20")
        if float(simi)==1.0:
            simi_eq2+=1
    reader.close()
    print(str(len(label_lst)))
    print("COMPARE WITH ZINC: max:{:.2f};min:{:.2f};average:{:.2f};median:{:.2f}\n".format(max(group_simi[len3:]),min(group_simi[len3:]),mean(group_simi[len3:]),median(group_simi[len3:])))
    print("Number of molecules with a similarity of 1 to ZINC20:{}\n".format(simi_eq2))
    print(f"plot number:{len(group_simi)}")
    
    ### Plot
    fig=FIG.add_subplot(sub_lst[rdx])
    sns.violinplot(label_lst,group_simi,palette="Set2",linewidth=3)
    fig.set_xticks([])
    fig.set_title(stdname,size=90)
    fig.tick_params(labelsize=80)
    fig.set_ylim([0,1])
    if not rdx in [0,4]:
        fig.set_yticks([])
FIG.text(0.07,0.35,"Tanimoto Similarity",size=100,ha="center",rotation='vertical')
legend_elements =[Patch(facecolor=current_palette[0], edgecolor='k',linewidth=3,
                         label="ChEMBL"),
                    Patch(facecolor=current_palette[1], edgecolor='k',linewidth=3,
                         label="ChemBridge"),
                    Patch(facecolor=current_palette[2], edgecolor='k',linewidth=3,
                         label="ZINC20")]
FIG.legend(handles=legend_elements,loc="center",prop={'size':100},bbox_to_anchor=(0.5,0.97),ncol=3,frameon=False)
plt.savefig("../result_graphs/figure5_TC_similarity.png")