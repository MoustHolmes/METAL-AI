import torch

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
                torch.zeros(max_csf_count - original_length, 2),  # change the 2 to the number of allowed excitations
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