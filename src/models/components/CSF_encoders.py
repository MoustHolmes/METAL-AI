import torch
from torch import nn

class simple_CSF_encoder(nn.Module):
    def __init__(self, output_size=32):
        super(simple_CSF_encoder, self).__init__()
        # Assuming the input size is 6 because we append n_electrons (1) and n_protons (1) to each 4-dimensional CSF.
        self.output_size = output_size
        self.network = nn.Sequential(
        nn.Linear(4, 64), # number of allowed excitations + 2
        nn.ReLU(),
        nn.Linear(64, output_size)
        )

    def forward(self, input_dict):
        excitations = input_dict["excitations"]
        n_electrons = input_dict["n_electrons"]
        n_protons = input_dict["n_protons"]

        # Append n_electrons and n_protons to each excitations
        n_electrons = n_electrons.float().unsqueeze(-1).unsqueeze(-1).expand(-1, excitations.size(1), 1)
        n_protons = n_protons.float().unsqueeze(-1).unsqueeze(-1).expand(-1, excitations.size(1), 1)
        extended_excitations = torch.cat([excitations, n_electrons, n_protons], dim=-1)
        return self.network(extended_excitations)
    
class no_CSF_encoder(nn.Module):
    def __init__(self,):
        super(simple_CSF_encoder, self).__init__()
        # Assuming the input size is 6 because we append n_electrons (1) and n_protons (1) to each 4-dimensional CSF.


    def forward(self, input_dict):

        excitations = input_dict["excitations"]
        n_electrons = input_dict["n_electrons"]
        n_protons = input_dict["n_protons"]

        # Append n_electrons and n_protons to each CSF
        
        n_electrons = n_electrons.float().unsqueeze(-1).unsqueeze(-1).expand(-1, excitations.size(1), 1)
        n_protons = n_protons.float().unsqueeze(-1).unsqueeze(-1).expand(-1, excitations.size(1), 1)
        extended_excitations = torch.cat([excitations, n_electrons, n_protons], dim=-1)
        return extended_excitations

class RotaryEmbedding2Angle4D(nn.Module):
    def __init__(self, dim):
        """
        Initialize the RepeatedRotaryEmbedding module.

        Args:
            dim (int): Dimension of the embeddings, must be divisible by 4.
        """
        super().__init__()
        assert dim % 4 == 0, "Embedding dimension must be divisible by 4"
        self.dim = dim

    def forward(self, embeddings, theta, phi):
        """
        Applies the repeated rotation on the embeddings using angles theta and phi.

        Args:
            embeddings (torch.Tensor): Input embeddings of shape [batch_size, n, dim].
            theta (torch.Tensor): Rotation angles for xy planes of shape [batch_size].
            phi (torch.Tensor): Rotation angles for zw planes of shape [batch_size].

        Returns:
            torch.Tensor: Rotated embeddings of the same shape as input.
        """
        batch_size, n, dim = embeddings.shape
        assert dim == self.dim, f"Input embedding dimension {dim} does not match initialized dimension {self.dim}"

        num_blocks = dim // 4  # Number of 4x4 blocks per embedding

        # Compute rotation components
        cos_theta = torch.cos(theta).unsqueeze(1).repeat(1, num_blocks)  # Shape: [batch_size, num_blocks]
        sin_theta = torch.sin(theta).unsqueeze(1).repeat(1, num_blocks)  # Shape: [batch_size, num_blocks]
        cos_phi = torch.cos(phi).unsqueeze(1).repeat(1, num_blocks)      # Shape: [batch_size, num_blocks]
        sin_phi = torch.sin(phi).unsqueeze(1).repeat(1, num_blocks)      # Shape: [batch_size, num_blocks]

        # Create block-diagonal rotation matrix for the entire batch
        R = torch.zeros(batch_size, dim, dim, device=embeddings.device)

        for i in range(num_blocks):
            # Define indices for the i-th block
            start = i * 4
            end = start + 4

            # Fill the block-diagonal matrix for all batches
            R[:, start:start+2, start:start+2] = torch.stack([
                torch.stack([cos_theta[:, i], -sin_theta[:, i]], dim=-1),
                torch.stack([sin_theta[:, i], cos_theta[:, i]], dim=-1)
            ], dim=1)

            R[:, start+2:end, start+2:end] = torch.stack([
                torch.stack([cos_phi[:, i], -sin_phi[:, i]], dim=-1),
                torch.stack([sin_phi[:, i], cos_phi[:, i]], dim=-1)
            ], dim=1)

        # Apply the rotation to embeddings
        rotated_embeddings = torch.einsum('bnd,bdm->bnm', embeddings, R)

        return rotated_embeddings

class PairEmbedding(nn.Module):
    def __init__(self, embedding_dim, pair_list=None):
        """
        Args:
            embedding_dim (int): Dimension of the embedding.
            pair_list (list of tuples): List of unique (x, y) pairs to index.
        """
        super(PairEmbedding, self).__init__()
        if pair_list is None:
            pair_list = [
                (0, 0), (1, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 1),
                (0, 3), (2, 2), (1, 3), (0, 4), (0, 5), (3, 2), (1, 4),
                (2, 3), (3, 3), (2, 4), (1, 5), (3, 4), (4, 3), (2, 5),
                (3, 5), (4, 4), (5, 4), (4, 5), (5, 5)
            ]
        
        # Initialize the embedding layer
        self.embedding = nn.Embedding(len(pair_list), embedding_dim)
        
        # Create the lookup table
        self.max_val = max(max(pair) for pair in pair_list)
        lookup_table = torch.full((self.max_val + 1, self.max_val + 1), -1, dtype=torch.long)
        for idx, pair in enumerate(pair_list):
            lookup_table[pair[0], pair[1]] = idx
        
        # Register the lookup table as a buffer so it moves with the module
        self.register_buffer("lookup_table", lookup_table, persistent=True)

    def forward(self, pair_tensor):
        """
        Args:
            pair_tensor (torch.Tensor): Tensor of shape [batch_size, n, 2] containing pairs.

        Returns:
            torch.Tensor: Embedding tensor of shape [batch_size, n, embedding_dim].
        """
        # Use the lookup table to find indices for all pairs in the batch
        indices = self.lookup_table[pair_tensor[..., 0], pair_tensor[..., 1]]
        
        # Pass the indices to the embedding layer
        embedded = self.embedding(indices)  # This will be of shape [batch_size, n, embedding_dim]
        return embedded



class ExcitaionEmbeddingIonRoPE(nn.Module):
    def __init__(self, output_size, angle_scale=0.05):
        super(ExcitaionEmbeddingIonRoPE, self).__init__()
        self.output_size = output_size
        self.angle_scale = angle_scale

        self.excitation_embedding = PairEmbedding(embedding_dim=output_size)
        self.ion_rope = RotaryEmbedding2Angle4D(dim=output_size)

    def forward(self, input_dict):
        excitations = input_dict["excitations"]
        n_electrons = input_dict["n_electrons"]
        n_protons = input_dict["n_protons"]


        # Embed the excitations
        embedded_excitations = self.excitation_embedding(excitations)

        # rotary positional encoding for the ions
        embedding = self.ion_rope(embedded_excitations, n_electrons * self.angle_scale, n_protons * self.angle_scale)

        return embedding
