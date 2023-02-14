import pickle

from typing import List

from model_container import ModelContainer
from scoring.component_parameters import ComponentParameters
from scoring.score_components.base_score_component import BaseScoreComponent
from scoring.score_summary import ComponentSummary

import torch
from mpn_models import mpnn, utils
from joblib import dump, load
from mpn_models.dmpnn import MPNN
from numpy import mean
import numpy as np

from rdkit.Chem import AllChem as Chem
from scoring.score_transformations import TransformationFactory
from utils.enums.transformation_type_enum import TransformationTypeEnum
from running_modes.reinforcement_learning.memorymonitor import MemoryMonitor
from tqdm import tqdm, trange
import ray
from ray.util.sgd import TorchTrainer

class CDockScore(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self.activity_model = self._load_dock_model(parameters)
        self._transformation_function = self._assign_transformation(self.parameters.specific_parameters)

    def calculate_score(self, molecules: List) -> ComponentSummary:

        smi_lst = []
        for m in molecules:
            smi = Chem.MolToSmiles(m)
            smi_lst.append(smi) 
        #try:    
        score_pre = self.predict(self.activity_model, smi_lst, scaler=self.scaler,batch_size=len(smi_lst)) 
        score = self._transformation_function(score_pre, self.parameters.specific_parameters)
        # except:
        #     score=[0 for i in range(len(smi_lst))]
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters)
        return score_summary

    def _load_dock_model(self, parameters: ComponentParameters):
        ckpt = parameters.model_path
        scaler=parameters.model_path.split("checkpoints/epoch=")[0]+"std_scaler.bin"
        self.scaler = load(scaler)
        self._predict = ray.remote(num_cpus=1, num_gpus=1)(mpnn.predict) 
        self.use_gpu = ray.cluster_resources().get('GPU', 0) > 0
        ###load ckpt
        dict_pt = torch.load(ckpt)["state_dict"]
        new_dict = {}
        for k in dict_pt.keys():
            ki = k.split("mpnn.")[1]
            new_dict[ki] = dict_pt[k]
        
        ###load model
        mpnt = MPNN()
        mpnt.model.load_state_dict(new_dict)
        mpnt.model.eval() 
        return mpnt.model
    def _assign_transformation(self, specific_parameters: {}):
        transformation_type = TransformationTypeEnum()
        factory = TransformationFactory()
        if self.parameters.specific_parameters[self.component_specific_parameters.TRANSFORMATION]:
            transform_function = factory.get_transformation_function(specific_parameters)
        else:
            self.parameters.specific_parameters[
                self.component_specific_parameters.TRANSFORMATION_TYPE] = transformation_type.NO_TRANSFORMATION
            transform_function = factory.no_transformation
        return transform_function
    
    def predict(self,model,smis,scaler,batch_size,ncpu:int=1):
        """Generate predictions for the inputs xs"""
        smis_batches = utils.batches(smis, 10000)
        model = ray.put(model)
        scaler = ray.put(scaler)
        
        refs = [
            self._predict.remote(
                model, smis, batch_size, ncpu,
                True, scaler, self.use_gpu, True
            ) for smis in smis_batches
        ]
        preds_chunks = [
            ray.get(r) for r in tqdm(refs, desc='Prediction', leave=False)
        ]
        pred_scores=[]
        for idx in range(len(preds_chunks)):
            for pred_score in preds_chunks[idx][0]:
                pred_scores.append(pred_score[0])
        return pred_scores    