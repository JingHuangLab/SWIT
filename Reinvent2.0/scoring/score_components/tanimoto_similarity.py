import numpy as np
from rdkit.Chem import DataStructs
from typing import List
from rdkit.Chem import AllChem

from scoring.component_parameters import ComponentParameters
from scoring.score_components.base_score_component import BaseScoreComponent
from scoring.score_summary import ComponentSummary
from scoring.score_components.count_fans_number import fans_number_control

from scoring.score_transformations import TransformationFactory
from utils.enums.transformation_type_enum import TransformationTypeEnum

class TanimotoSimilarity(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self._fingerprints, _ = self._smiles_to_fingerprints(self.parameters.smiles)
        self._transformation_function = self._assign_transformation(self.parameters.specific_parameters)

    def calculate_score(self, molecules: List) -> ComponentSummary:
        query_fps = self._mols_to_fingerprints(molecules)
        score,ref_id = self._calculate_tanimoto(query_fps, self._fingerprints)  # query_fps is newborn smiles.self._fingerprints is inhibitor.smi.
        score = self._transformation_function(score, self.parameters.specific_parameters)
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters)
        return score_summary

    def _mols_to_fingerprints(self, molecules: List, radius=3, useCounts=True, useFeatures=True) -> []:
        
        fingerprints = [AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius,

        ) for mol in molecules]
        
        # fingerprints = [AllChem.GetMorganFingerprintAsBitVect(
        #     mol,
        #     radius,
            #         useCounts=useCounts,
            # useFeatures=useFeatures
        # ) 
        return fingerprints

    def _calculate_tanimoto(self, query_fps, ref_fingerprints) -> np.array:
        # return np.array([np.max(DataStructs.BulkTanimotoSimilarity(fp, ref_fingerprints)) for fp in query_fps])
        max_lst = []
        id_lst = []
        for fp in query_fps:
            simi_lst = [DataStructs.TanimotoSimilarity(fp, inhi) for inhi in ref_fingerprints]
            max_value= np.max(simi_lst)
            max_id = simi_lst.index(max_value)
            max_lst.append(max_value)
            id_lst.append(max_id)
        return max_lst,id_lst

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


