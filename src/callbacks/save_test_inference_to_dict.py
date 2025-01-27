
import torch
from lightning.pytorch.callbacks import Callback
import pickle
from typing import Tuple


class SaveTestInferenceToDict(Callback):
    """Callback for saving test inference to a dictionary.

    Args:
        save_dir (str): path to save the dictionary.

    Attributes:
        save_dir (str):  path to save the dictionary.
        pred_dict (Dict): Dict used to save predictions.
    """

    def __init__(self, save_dir: str, filename: str = 'results'):
        self.save_dir = save_dir
        self.filename = filename

    def on_test_start(self, trainer, pl_module):
        self.pred_dict = {}

    def on_test_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
        ):
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        mask = outputs['mask']
        preds = outputs['preds'][mask]
        targets = outputs['targets'][mask]
        
        
        # print(mask.shape)
        # print(csf.shape)
        # print(csf.unsqueeze(1).shape)
        
        # print(csf)
        excitations = batch["excitations"][mask]
        # print(excitations)
        # print(excitations[0])
        # print( mask.sum().item() )
        
        n = batch["converged"].shape[-1]
        
        # print(csf.repeat(n,1,1,1).shape)
        # print(csf.repeat(n,1,1,1).view(-1, n, 2)[mask.view(-1)].shape)
        # print(csf.repeat(n,1,1,1).view(-1, n, 2)[mask.view(-1)])
        # print(csf.repeat(n,1,1,1)[mask.unsqueeze(0).repeat(n, 1, 1)])
        n_protons = batch["n_protons"].unsqueeze(1).repeat(1,n)[mask]
        # print(csf.unsqueeze(1).repeat(1,1,n)[mask.unsqueeze(-1)])
        n_electrons = batch["n_electrons"].unsqueeze(1).repeat(1,n)[mask]
        csf = batch["excitations"].repeat(n,1,1,1).view(-1, n, 2)[mask.view(-1)]

        for i in range(len(targets)):
            ion_key = (n_protons[i].item(), n_electrons[i].item())
            # print(ion_key)
            csf_key = tuple(map(tuple, csf[i].cpu().numpy()))
            # print(csf_key)

            if ion_key not in self.pred_dict:
                self.pred_dict[ion_key] = {}
            
            if csf_key not in self.pred_dict[ion_key]:
                self.pred_dict[ion_key][csf_key] = {}

            self.pred_dict[ion_key][csf_key]['preds'] = preds[i].cpu().numpy()
            self.pred_dict[ion_key][csf_key]['targets'] = targets[i].cpu().numpy()
            self.pred_dict[ion_key][csf_key]['excitation'] = excitations[i].cpu().numpy()

    def on_test_end(self, trainer, pl_module):
        with open(self.save_dir+f'/{self.filename}.pkl', 'wb') as f:
            pickle.dump(self.pred_dict, f)


class GaussianNLLSaveTestInferenceToDict(Callback):
    """Callback for saving test inference to a dictionary.

    Args:
        save_dir (str): path to save the dictionary.

    Attributes:
        save_dir (str):  path to save the dictionary.
        pred_dict (Dict): Dict used to save predictions.
    """

    def __init__(self, save_dir: str, filename: str = 'results'):
        self.save_dir = save_dir
        self.filename = filename

    def on_test_start(self, trainer, pl_module):
        self.pred_dict = {}

    def on_test_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
        ):
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        mask = outputs['mask']
        mean = outputs['preds'][:,:,0][mask]
        variance = outputs['preds'][:,:,1][mask]
        # preds = outputs['preds'][mask]
        targets = outputs['targets'][mask]
        
        
        # print(mask.shape)
        # print(csf.shape)
        # print(csf.unsqueeze(1).shape)
        
        # print(csf)
        excitations = batch["excitations"][mask]
        # print(excitations)
        # print(excitations[0])
        # print( mask.sum().item() )
        
        n = batch["converged"].shape[-1]
        
        # print(csf.repeat(n,1,1,1).shape)
        # print(csf.repeat(n,1,1,1).view(-1, n, 2)[mask.view(-1)].shape)
        # print(csf.repeat(n,1,1,1).view(-1, n, 2)[mask.view(-1)])
        # print(csf.repeat(n,1,1,1)[mask.unsqueeze(0).repeat(n, 1, 1)])
        n_protons = batch["n_protons"].unsqueeze(1).repeat(1,n)[mask]
        # print(csf.unsqueeze(1).repeat(1,1,n)[mask.unsqueeze(-1)])
        n_electrons = batch["n_electrons"].unsqueeze(1).repeat(1,n)[mask]
        csf = batch["excitations"].repeat(n,1,1,1).view(-1, n, 2)[mask.view(-1)]

        for i in range(len(mean)):
            ion_key = (n_protons[i].item(), n_electrons[i].item())
            # print(ion_key)
            csf_key = tuple(map(tuple, csf[i].cpu().numpy()))
            # print(csf_key)

            if ion_key not in self.pred_dict:
                self.pred_dict[ion_key] = {}
            
            if csf_key not in self.pred_dict[ion_key]:
                self.pred_dict[ion_key][csf_key] = {}

            self.pred_dict[ion_key][csf_key]['mean'] = mean[i].cpu().numpy()
            self.pred_dict[ion_key][csf_key]['variance'] = variance[i].cpu().numpy()
            self.pred_dict[ion_key][csf_key]['targets'] = targets[i].cpu().numpy()
            self.pred_dict[ion_key][csf_key]['excitation'] = excitations[i].cpu().numpy()

    def on_test_end(self, trainer, pl_module):
        with open(self.save_dir+f'/{self.filename}.pkl', 'wb') as f:
            pickle.dump(self.pred_dict, f)
