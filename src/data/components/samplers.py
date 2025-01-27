from torch.utils.data import Sampler
import random

# class GroupedBatchSampler(Sampler):
#     def __init__(self, dataset, batch_size, shuffle=True):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.grouped_indices = dataset.grouped_indices
#         if shuffle:
#             self.group_lengths = sorted(self.grouped_indices.keys())
#         else:
#             self.group_lengths = list(self.grouped_indices.keys())

#     def __iter__(self):
#         for length in self.group_lengths:
#             indices = self.grouped_indices[length]
#             random.shuffle(indices)
#             # Yield full batches only
#             for i in range(0, len(indices), self.batch_size):
#                 if i + self.batch_size <= len(indices):
#                     yield indices[i:i + self.batch_size]

#     def __len__(self):
#         # This is an approximation, as we drop the last incomplete batch in each group
#         return sum(len(indices) // self.batch_size for indices in self.grouped_indices.values())

class GroupedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.grouped_indices = dataset.grouped_indices
        self.shuffle = shuffle

    def __iter__(self):
        all_batches = []

        # Sort the group lengths if shuffle is False
        group_lengths = sorted(self.grouped_indices.keys()) if not self.shuffle else list(self.grouped_indices.keys())

        for length in group_lengths:
            indices = self.grouped_indices[length]
            # Shuffle within each group if shuffle is True
            if self.shuffle:
                random.shuffle(indices)
            
            # Form batches from the indices of this group
            for i in range(0, len(indices), self.batch_size):
                if i + self.batch_size <= len(indices):
                    all_batches.append(indices[i:i + self.batch_size])

        # Shuffle all batches if shuffle is True
        if self.shuffle:
            random.shuffle(all_batches)

        # Yield each batch one by one
        for batch in all_batches:
            yield batch

    def __len__(self):
        # This is an approximation, as we drop the last incomplete batch in each group
        return sum(len(indices) // self.batch_size for indices in self.grouped_indices.values())
