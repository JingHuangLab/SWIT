from re import X
from typing import Iterable, Optional

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from ..chemprop.data import (
    StandardScaler, MoleculeDataLoader, MoleculeDataset, MoleculeDatapoint
)

def predict(model, smis: Iterable[str], batch_size: int = 50, ncpu: int = 1, 
            uncertainty: bool = False, scaler: Optional[StandardScaler] = None,
            use_gpu: bool = False, disable: bool = False):
    """Predict the target values of the given SMILES strings with the input 
    model

    Parameters
    ----------
    model : mpnn.MoleculeModel
        the model to use
    smis : Iterable[str]
        the SMILES strings to perform inference on
    batch_size : int, default=50
        the size of each minibatch (impacts performance)
    ncpu : int, default=1
        the number of cores over which to parallelize input preparation
    uncertainty : bool, default=False
        whether the model predicts its own uncertainty
    scaler : StandardScaler, default=None
        A StandardScaler object fit on the training targets. If none,
        prediction values will not be transformed to original dataset
    use_gpu : bool, default=False
        whether to use the GPU during inference
    disable : bool, default=False
        whether to disable the progress bar

    Returns
    -------
    predictions : np.ndarray
        an NxM array where N is the number of inputs for which to produce 
        predictions and M is the number of prediction tasks
    """
    device = 'cuda' if use_gpu else 'cpu'
    model.to(device)

    dataset = MoleculeDataset([MoleculeDatapoint([smi]) for smi in smis])
    data_loader = MoleculeDataLoader(
        dataset=dataset, batch_size=batch_size,
        num_workers=ncpu, pin_memory=use_gpu
    )
    model.eval()

    pred_batches = []
    with torch.no_grad():
        # pred_batches = [
        #     model(batch_graph)
        #     for batch_graph, _ in tqdm(
        #         data_loader, desc='Inference', unit='batch', leave=False, disable=disable
        #     )
        # ]
        for batch in tqdm(data_loader, desc='Inference', unit='batch',
                          leave=False, disable=disable):
            componentss, _ = batch#.batch_graph()
            componentss = [
                [X.to(device)#, non_blocking=True)
                 if isinstance(X, torch.Tensor) else X for X in components]
                for components in componentss
            ]
            pred_batch = model(componentss)
            pred_batches.append(pred_batch)#.data.cpu().numpy())

        preds = torch.cat(pred_batches)
    # preds = np.concatenate(pred_batches)
    preds = preds.cpu().numpy()

    if uncertainty:
        means = preds[:, 0::2]
        variances = preds[:, 1::2]

        if scaler:
            means = scaler.inverse_transform(means)
            variances = scaler.stds**2 * variances

        return means, variances

    if scaler:
        preds = scaler.inverse_transform(preds)

    return preds

# @ray.remote(num_cpus=ncpu)
# def predict(model, smis, batch_size, ncpu, scaler, use_gpu: bool):
#     model.device = 'cpu'

#     test_data = MoleculeDataset(
#         [MoleculeDatapoint(smiles=[smi]) for smi in smis]
#     )
#     data_loader = MoleculeDataLoader(
#         dataset=test_data, batch_size=batch_size, num_workers=ncpu
#     )
#     return _predict(
#         model, data_loader, self.uncertainty,
#         disable=True, scaler=scaler
#     )

def _predict(model: nn.Module, data_loader: Iterable, uncertainty: bool,
            disable: bool = False,
            scaler: Optional[StandardScaler] = None) -> np.ndarray:
    """Predict the output values of a dataset

    Parameters
    ----------
    model : nn.Module
        the model to use
    data_loader : MoleculeDataLoader
        an iterable of MoleculeDatasets
    uncertainty : bool
        whether the model predicts its own uncertainty
    disable : bool (Default = False)
        whether to disable the progress bar
    scaler : Optional[StandardScaler] (Default = None)
        A StandardScaler object fit on the training targets

    Returns
    -------
    predictions : np.ndarray
        an NxM array where N is the number of inputs for which to produce 
        predictions and M is the number of prediction tasks
    """
    model.eval()

    pred_batches = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Inference', unit='batch',
                          leave=False, disable=disable):
            batch_graph = batch.batch_graph()
            pred_batch = model(batch_graph)
            pred_batches.append(pred_batch.data.cpu().numpy())
    preds = np.concatenate(pred_batches)

    if uncertainty:
        means = preds[:, 0::2]
        variances = preds[:, 1::2]

        if scaler:
            means = scaler.inverse_transform(means)
            variances = scaler.stds**2 * variances

        return means, variances

    # Inverse scale if regression
    if scaler:
        preds = scaler.inverse_transform(preds)

    return preds
    
