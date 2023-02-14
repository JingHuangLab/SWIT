from molvs import standardize_smiles as ss

class fans_number_control:

    ## class variables
    count_lst = []   
    temp_lst = []
    cnt_len = 0
    temp_smi = []

    @classmethod
    def initial_cnt_lst(cls,len_num):
        cls.count_lst = [0 for i in range(len_num)]
        cls.cnt_len = len_num

    ## class method
    @classmethod
    def save_cntlst_and_smiles(cls,idol_id,ts_smiles):
        cls.temp_lst = idol_id
        cls.temp_smi = ts_smiles

    @classmethod
    def update_cntlst_and_get_score(cls,score,all_smiles):
        score = score.tolist()
        score_lst = []
        inter_cnt = 0
        # print("length of temp_smiles:{},demo:{}".format(len(cls.temp_smi),cls.temp_smi[0]))
        # print("length of all smiles:{},demo:{}".format(len(all_smiles),ss(all_smiles[0])))

        for idx,sc in enumerate(score):
            if sc>=0.9:
                if ss(all_smiles[idx]) in cls.temp_smi:
                    lst_idx= cls.temp_smi.index(ss(all_smiles[idx]))
                    ref_id = cls.temp_lst[lst_idx]
                    inter_cnt+=1
                    # print("ref_id:{}".format(ref_id))
                    if cls.count_lst[ref_id]<=int(10000/cls.cnt_len):
                        cls.count_lst[ref_id] +=1
                        # print("count_lst +1")
                        score_lst.append(sc)
                    else:
                        score_lst.append(0)
                else:
                    score_lst.append(sc)
            else:
                score_lst.append(sc)
        # print("------------------------------------------------------------")
        # print("intersection between all and temp smiles:{}".format(inter_cnt))
        return score_lst

    @classmethod
    def update_cntlst(cls,score,all_smiles):
        for idx,sc in enumerate(score):
            if sc>=0.9:
                if ss(all_smiles[idx]) in cls.temp_smi:
                    lst_idx= cls.temp_smi.index(ss(all_smiles[idx]))
                    ref_id = cls.temp_lst[lst_idx]
                    cls.count_lst[ref_id] +=1


            
        
