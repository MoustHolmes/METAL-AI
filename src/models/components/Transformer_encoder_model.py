import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable, Optional


class simple_transformer_encoder_model(nn.Module):
    def __init__(
        self,
        csf_encoder,
        input_size: int = 4, # number of allowed excitations + 2
        d_model: int = 64,
        nhead: int = 8,
        dim_forward: int = 64,
        num_layers: int = 6,
        output_size: int =1,
        dropout: float = 0.5,
        output_activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super(simple_transformer_encoder_model, self).__init__()
        self.csf_encoder = csf_encoder
        encoder_layers = nn.TransformerEncoderLayer(
            self.csf_encoder.output_size, nhead, dim_forward, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(self.csf_encoder.output_size, output_size)
        self.output_activation = output_activation

    def forward(self, input_dict):
        encoded_csfs = self.csf_encoder(input_dict)
        output = self.transformer_encoder(encoded_csfs)#,src_key_padding_mask = input_dict['pad_mask']  src_key_padding_mask = ~input_dict['mask']
        # output = output[:, 0, :]
        output = self.decoder(output)

        if self.output_activation:
            output = self.output_activation(output)

        return output.squeeze(-1)


        # csfs = input_dict["excitations"]
        # encoded_csfs = self.csf_encoder(
        #     csfs, input_dict["n_electrons"], input_dict["n_protons"]
        # )
        # output = self.transformer_encoder(
        #     encoded_csfs, src_key_padding_mask=input_dict["pad_mask"]
        # )  # src_key_padding_mask = ~input_dict['mask']
        # # output = output[:, 0, :]
        # output = self.decoder(output)
        # return F.sigmoid(output).squeeze(-1)
