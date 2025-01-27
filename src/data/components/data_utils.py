import pickle
import random
import numpy as np

def load_data_dict(data_dir):
    # Load the data dictionary from the data directory
    with open(data_dir, "rb") as f:
        data_dict = pickle.load(f)

    return data_dict

def save_data_dict(data_dict, data_dir):
    # Save the data dictionary to the data directory
    with open(data_dir, "wb") as f:
        pickle.dump(data_dict, f)

def top_asf_size_train_val_split(data_dict, validation_percentage=0.05):
    # Flatten the data_dict to a list of tuples: (ion_key, asf_key, excitations_length)
    flat_data = [
        (ion_key, asf_key, len(data_dict[ion_key][asf_key]["excitations"]))
        for ion_key in data_dict.keys()
        for asf_key in data_dict[ion_key].keys()
    ]

    # Sort by the length of excitations in descending order
    flat_data.sort(key=lambda x: x[2], reverse=True)

    # Determine the number of validation data points
    validation_size = int(len(flat_data) * validation_percentage)

    # Create the validation and training datasets
    validation_data = {}
    training_data = {}

    # Add the top 5% to the validation set
    for ion_key, asf_key, _ in flat_data[:validation_size]:
        if ion_key not in validation_data:
            validation_data[ion_key] = {}
        validation_data[ion_key][asf_key] = data_dict[ion_key][asf_key]

    # Add the remaining 95% to the training set
    for ion_key, asf_key, _ in flat_data[validation_size:]:
        if ion_key not in training_data:
            training_data[ion_key] = {}
        training_data[ion_key][asf_key] = data_dict[ion_key][asf_key]

    return training_data, validation_data

class TopASFSizeTrainValSplitter:
    def __init__(self, validation_percentage: float = 0.05):
        self.validation_percentage = validation_percentage

    def __call__(self, data_dict):
        return top_asf_size_train_val_split(data_dict, self.validation_percentage)

def random_train_val_split(data_dict, validation_percentage=0.05):
    # Flatten the data_dict to a list of tuples: (ion_key, asf_key)
    flat_data = [
        (ion_key, asf_key)
        for ion_key in data_dict.keys()
        for asf_key in data_dict[ion_key].keys()
    ]
    


    
    # Shuffle the data randomly
    random.shuffle(flat_data)
    
    # Determine the number of validation data points
    validation_size = int(len(flat_data) * validation_percentage)
    
    # Create the validation and training datasets
    validation_data = {}
    training_data = {}
    
    # Add the first part to the validation set
    for ion_key, asf_key in flat_data[:validation_size]:
        if ion_key not in validation_data:
            validation_data[ion_key] = {}
        validation_data[ion_key][asf_key] = data_dict[ion_key][asf_key]
    
    # Add the remaining part to the training set
    for ion_key, asf_key in flat_data[validation_size:]:
        if ion_key not in training_data:
            training_data[ion_key] = {}
        training_data[ion_key][asf_key] = data_dict[ion_key][asf_key]
    
    return training_data, validation_data

class RandomTrainValSplitter:
    def __init__(self, validation_percentage: float = 0.05):
        self.validation_percentage = validation_percentage

    def __call__(self, data_dict):
        return random_train_val_split(data_dict, self.validation_percentage)

def triple_train_val_split(
    data_dict, 
    validation_percentage=0.05,
    ASF_size_percentage=0.05,
    include_ion=None,
    unseen_ion=None,
    remove_nan_effect=False,
):
    """
    Split data into:
    - `training_data`: Remaining training data after validation splits.
    - `validation_data`: Random subset for validation from training ions.
    - `unseen_data`: Data belonging to unseen ions (not part of include_ion).
    - `large_data`: ASFs larger than the calculated thresholds.
    """

    include_ion = [tuple(item) for item in include_ion] if include_ion else None
    unseen_ion = [tuple(item) for item in unseen_ion] if unseen_ion else None

    # Step 1: Calculate thresholds for large ASFs for each ion
    ion_threshold = {}
    for ion_key in data_dict.keys():
        # Get all ASF sizes for this ion, applying `remove_nan_effect` if True
        asf_sizes = [
            len(asf_data["excitations"])
            for asf_key, asf_data in data_dict[ion_key].items()
            if not (remove_nan_effect and ("effect" not in asf_data or np.isnan(asf_data["effect"]).all()))
        ]
        if not asf_sizes:  # If no valid ASFs remain, skip this ion
            ion_threshold[ion_key] = float('inf')  # No ASFs qualify
            continue
        # Sort ASF sizes in descending order
        asf_sizes.sort(reverse=True)
        # Calculate the threshold index
        threshold_index = max(0, int(len(asf_sizes) * ASF_size_percentage) - 1)
        # Determine the threshold size
        ion_threshold[ion_key] = asf_sizes[threshold_index]

    # Step 2: Initialize splits
    training_data = {}
    validation_data = {}
    unseen_data = {}
    large_data = {}

    # Step 3: Assign data points to the correct sets
    for ion_key, asfs in data_dict.items():
        for asf_key, asf_data in asfs.items():
            if remove_nan_effect:
                    if "effect" not in asf_data:
                        continue
                    if np.isnan(asf_data["effect"]).all():
                        continue
            asf_size = len(asf_data["excitations"])
            above_threshold = asf_size >= ion_threshold[ion_key]

            # Check if ion belongs to unseen_ion
            if unseen_ion is not None and ion_key in unseen_ion:
                if not above_threshold:
                    if ion_key not in unseen_data:
                        unseen_data[ion_key] = {}
                    unseen_data[ion_key][asf_key] = asf_data
            # Include ions in training or validation
            elif include_ion is None or ion_key in include_ion:
                if above_threshold:
                    if ion_key not in large_data:
                        large_data[ion_key] = {}
                    large_data[ion_key][asf_key] = asf_data
                else:
                    if ion_key not in training_data:
                        training_data[ion_key] = {}
                    training_data[ion_key][asf_key] = asf_data

    # Step 4: Randomly split training_data into training and validation subsets
    flat_training_data = [
        (ion_key, asf_key)
        for ion_key, asfs in training_data.items()
        for asf_key in asfs.keys()
    ]
    random.shuffle(flat_training_data)
    validation_size = int(len(flat_training_data) * validation_percentage)
    validation_subset = flat_training_data[:validation_size]

    final_training_data = {}
    for ion_key, asf_key in flat_training_data[validation_size:]:
        if ion_key not in final_training_data:
            final_training_data[ion_key] = {}
        final_training_data[ion_key][asf_key] = training_data[ion_key][asf_key]

    for ion_key, asf_key in validation_subset:
        if ion_key not in validation_data:
            validation_data[ion_key] = {}
        validation_data[ion_key][asf_key] = training_data[ion_key][asf_key]

    return final_training_data, validation_data, unseen_data, large_data

class TripleTrainValSplitter:
    def __init__(
        self,
        validation_percentage: float = 0.05,
        ASF_size_percentage: float = 0.05,
        include_ion=None,
        unseen_ion=None,
        remove_nan_effect=False,
    ):
        self.validation_percentage = validation_percentage
        self.ASF_size_percentage = ASF_size_percentage
        self.include_ion = include_ion
        self.unseen_ion = unseen_ion
        self.remove_nan_effect = remove_nan_effect

    def __call__(self, data_dict):
        return triple_train_val_split(
            data_dict,
            self.validation_percentage,
            self.ASF_size_percentage,
            self.include_ion,
            self.unseen_ion,
            self.remove_nan_effect,
        )

