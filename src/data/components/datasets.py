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
    def __init__(self, data_dict, remove_nan_effect=False, ion_include=None, ion_exclude=None):
        self.data = data_dict
        self.ion_include = ion_include
        self.ion_exclude = ion_exclude
        self.index_mapping = self._create_index_mapping(remove_nan_effect)
        self.grouped_indices = self._group_by_excitations_length()

    def _create_index_mapping(self, remove_nan_effect):
        mapping = []
        for ion_key in self.data.keys():
            # Apply inclusion and exclusion filters
            if self.ion_include is not None and ion_key not in self.ion_include:
                continue
            if self.ion_exclude is not None and ion_key in self.ion_exclude:
                continue

            for asf_key in self.data[ion_key].keys():
                data_point = self.data[ion_key][asf_key]

                if len(data_point["excitations"]) == 1:
                    continue

                data_point["filling_numbers"] = torch.tensor(asf_key)

                if not isinstance(data_point["excitations"], torch.Tensor):
                    data_point["excitations"] = torch.tensor(data_point["excitations"])
                
                if not isinstance(data_point["converged"], torch.Tensor):
                    data_point["converged_mask"] = torch.from_numpy(~np.isnan(data_point["converged"]))
                    data_point["converged"] = torch.tensor(data_point["converged"], dtype=torch.bool)

                if not isinstance(data_point["effect"], torch.Tensor):
                    data_point["effect"] = torch.tensor(data_point["effect"], dtype=torch.float32)

                data_point["n_protons"] = torch.tensor(ion_key[0], dtype=torch.long)
                data_point["n_electrons"] = torch.tensor(ion_key[1], dtype=torch.long)

                if remove_nan_effect:
                    if "effect" not in data_point:
                        continue
                    if np.isnan(data_point["effect"]).all():
                        continue
                mapping.append((ion_key, asf_key))

        return mapping

    def _group_by_excitations_length(self):
        groups = {}
        for idx, (ion_key, asf_key) in enumerate(self.index_mapping):
            excitations_length = len(self.data[ion_key][asf_key]["excitations"])
            if excitations_length == 1:
                continue
            if excitations_length not in groups:
                groups[excitations_length] = []
            groups[excitations_length].append(idx)
        return groups

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        ion_key, asf_key = self.index_mapping[idx]
        data_point = self.data[ion_key][asf_key]

        selected_data = {
            'excitations': data_point['excitations'],
            'filling_numbers': data_point['filling_numbers'],
            'converged': data_point['converged'],
            'converged_mask': data_point['converged_mask'],
            'effect': data_point['effect'],
            'n_protons': data_point['n_protons'],
            'n_electrons': data_point['n_electrons'],
        }
        return selected_data

        # return data_point
        # return {
        #     "ion_key": ion_key,
        #     "asf_key": asf_key,
        #     **data_point,  # Unpacking the data point into the returned dictionary
        # }
