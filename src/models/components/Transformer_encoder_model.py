import torch
from torch import nn
import torch.nn.functional as F


class simple_transformer_encoder_model(nn.Module):
    def __init__(
        self,
        csf_encoder,
        input_size: int = 4, # number of allowed excitations + 2
        d_model: int = 64,
        nhead: int = 8,
        dim_forward: int = 64,
        num_layers: int = 6,
        output_size=1,
        dropout=0.5,
    ):
        super(simple_transformer_encoder_model, self).__init__()
        self.csf_encoder = csf_encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_forward, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, output_size)

    def forward(self, input_dict):
        csfs = input_dict['excitations']
        encoded_csfs = self.csf_encoder(csfs, input_dict['n_electrons'], input_dict['n_protons'])
        output = self.transformer_encoder(encoded_csfs)#,src_key_padding_mask = input_dict['pad_mask']  src_key_padding_mask = ~input_dict['mask']
        # output = output[:, 0, :]
        output = self.decoder(output)
        return F.sigmoid(output).squeeze(-1)


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
