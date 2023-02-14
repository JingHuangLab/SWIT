from rdkit.Chem.Descriptors import MolLogP
from scoring.component_parameters import ComponentParameters
from scoring.score_components.physchem.base_physchem_component import BasePhysChemComponent


class SlogP(BasePhysChemComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)

    def _calculate_phys_chem_property(self, mol):
        ## origional
        return MolLogP(mol)

        # ##by zzz
        # logp = MolLogP(mol)
        # if logp>3:
        #     return 0
        # elif logp<0:
        #     return 0
        # else:
        #     return logp/(-3)+1
        # ##by zzz
