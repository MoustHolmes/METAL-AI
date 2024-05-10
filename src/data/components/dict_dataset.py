import torch
import h5py
import numpy as np
import scipy.spatial

from torch.utils.data.dataset import Dataset


def dict_collate_fn(batch):
    # Find the maximum number of 'CSFs' across all samples in the batch.
    max_csf_count = max([len(item["excitations"]) for item in batch])

    # Initialize lists to hold padded 'CSFs' and the original lengths.
    original_lengths = []

    padded_csfs_list = []
    padded_converged_list = []
    padded_converged_mask_list = []
    padded_effect_list = []

    for item in batch:
        original_length = item["excitations"].size(0)
        original_lengths.append(original_length)
        padded_csfs = torch.cat(
            [
                item["excitations"],
                torch.zeros(max_csf_count - original_length, 4),  # Pad with zeros.
            ]
        )
        # print(item['converged'])
        padded_converged = torch.cat(
            [
                item["converged"],
                torch.zeros(
                    max_csf_count - original_length,
                ),  # Pad with zeros. dtype=torch.bool
            ]
        )
        padded_converged_mask = torch.cat(
            [
                item["converged_mask"],
                torch.zeros(
                    max_csf_count - original_length,
                ),  # Pad with zeros. dtype=torch.bool
            ]
        )

        padded_effect = torch.cat(
            [
                item["effect"],
                torch.zeros(max_csf_count - original_length),  # Pad with zeros.
            ]
        )
        padded_csfs_list.append(padded_csfs)
        padded_converged_list.append(padded_converged)
        padded_converged_mask_list.append(padded_converged_mask)
        padded_effect_list.append(padded_effect)

    # Stack the padded 'CSFs'.
    padded_csfs = torch.stack(padded_csfs_list)
    padded_converged = torch.stack(padded_converged_list)
    padded_converged_mask = torch.stack(padded_converged_mask_list)
    padded_effect = torch.stack(padded_effect_list)

    # Create a mask based on the original lengths.
    mask = torch.ones_like(padded_csfs[:, :, 0], dtype=torch.bool)
    for i, length in enumerate(original_lengths):
        mask[i, :length] = False

    # Convert other attributes to tensors.
    n_electrons = torch.stack([item["n_electrons"] for item in batch])
    n_protons = torch.stack([item["n_protons"] for item in batch])

    # Return a dictionary with the batched data and the mask.
    return {
        "excitations": padded_csfs,
        "pad_mask": mask,
        "n_electrons": n_electrons,
        "n_protons": n_protons,
        "converged": padded_converged,
        "converged_mask": padded_converged_mask,
        "effect": padded_effect,
    }


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
