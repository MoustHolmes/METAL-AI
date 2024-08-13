import torch
import h5py
import numpy as np
import scipy.spatial

from torch.utils.data.dataset import Dataset

class SimpleDictDataset(Dataset):
    def __init__(self, data_dict, remove_nan_effect=False):
        """
        data: The nested dictionary containing all your data
        """
        self.data = data_dict
        self.index_mapping = self._create_index_mapping(remove_nan_effect)

    def _create_index_mapping(self, remove_nan_effect):
        mapping = []
        for ion_key in self.data.keys():
            for asf_key in self.data[ion_key].keys():

                data_point = self.data[ion_key][asf_key]

                data_point["excitations"] = torch.tensor(data_point["excitations"])
                data_point["converged_mask"] = torch.from_numpy(
                    ~np.isnan(data_point["converged"])
                )
                data_point["converged"] = torch.tensor(
                    data_point["converged"], dtype=torch.bool
                )

                if "effect" not in data_point:
                    data_point["effect"] = torch.full(
                        (len(data_point["excitations"]),), float("nan")
                    )
                else:
                    data_point["effect"] = torch.tensor(data_point["effect"])

                data_point["n_protons"] = torch.tensor(ion_key[0], dtype=torch.long)
                data_point["n_electrons"] = torch.tensor(ion_key[1], dtype=torch.long)

                if remove_nan_effect:
                    if "effect" not in self.data[ion_key][asf_key]:
                        continue
                    if np.isnan(self.data[ion_key][asf_key]["effect"]).all():
                        continue
                mapping.append((ion_key, asf_key))

        return mapping

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        # Retrieve the actual keys from the index
        ion_key, asf_key = self.index_mapping[idx]

        data_point = self.data[ion_key][asf_key]

        return data_point

class GroupedDictDataset(Dataset):
    def __init__(self, data_dict, remove_nan_effect=False):
        self.data = data_dict
        self.index_mapping = self._create_index_mapping(remove_nan_effect)
        self.grouped_indices = self._group_by_excitations_length()

    def _create_index_mapping(self, remove_nan_effect):
        mapping = []
        for ion_key in self.data.keys():
            for asf_key in self.data[ion_key].keys():
                data_point = self.data[ion_key][asf_key]
                # print(data_point)
                if not isinstance(data_point["excitations"], torch.Tensor):
                    data_point["excitations"] = torch.tensor(data_point["excitations"])
                
                if not isinstance(data_point["converged"], torch.Tensor):
                    data_point["converged_mask"] = torch.from_numpy(~np.isnan(data_point["converged"]))
                    data_point["converged"] = torch.tensor(data_point["converged"], dtype=torch.bool)

                if "effect" not in data_point:
                    data_point["effect"] = torch.full((len(data_point["excitations"]),), float("nan"))
                elif not isinstance(data_point["effect"], torch.Tensor):
                    data_point["effect"] = torch.tensor(data_point["effect"])

                # if not isinstance(data_point["n_protons"], torch.Tensor):
                data_point["n_protons"] = torch.tensor(ion_key[0], dtype=torch.long)
                
                # if not isinstance(data_point["n_electrons"], torch.Tensor):
                data_point["n_electrons"] = torch.tensor(ion_key[1], dtype=torch.long)

                if remove_nan_effect:
                    if "effect" not in self.data[ion_key][asf_key]:
                        continue
                    if np.isnan(self.data[ion_key][asf_key]["effect"]).all():
                        continue
                mapping.append((ion_key, asf_key))

        return mapping


    def _group_by_excitations_length(self):
        groups = {}
        for idx, (ion_key, asf_key) in enumerate(self.index_mapping):
            excitations_length = len(self.data[ion_key][asf_key]["excitations"])
            if excitations_length not in groups:
                groups[excitations_length] = []
            groups[excitations_length].append(idx)
        return groups

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        ion_key, asf_key = self.index_mapping[idx]
        data_point = self.data[ion_key][asf_key]
        return data_point
        # return {
        #     "ion_key": ion_key,
        #     "asf_key": asf_key,
        #     **data_point,  # Unpacking the data point into the returned dictionary
        # }