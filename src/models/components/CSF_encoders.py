import torch
from torch import nn

class simple_CSF_encoder(nn.Module):
    def __init__(self, output_size=32):
        super(simple_CSF_encoder, self).__init__()
        # Assuming the input size is 6 because we append n_electrons (1) and n_protons (1) to each 4-dimensional CSF.
        self.network = nn.Sequential(
        nn.Linear(4, 64), # number of allowed excitations + 2
        nn.ReLU(),
        nn.Linear(64, output_size)
        )

    def forward(self, csfs, n_electrons, n_protons):
        # Append n_electrons and n_protons to each CSF
        n_electrons = n_electrons.float().unsqueeze(-1).unsqueeze(-1).expand(-1, csfs.size(1), 1)
        n_protons = n_protons.float().unsqueeze(-1).unsqueeze(-1).expand(-1, csfs.size(1), 1)
        extended_csfs = torch.cat([csfs, n_electrons, n_protons], dim=-1)
        return self.network(extended_csfs)