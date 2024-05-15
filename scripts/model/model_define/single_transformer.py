import math
from collections import OrderedDict
from typing import Any, Callable, Dict, List, NamedTuple, Optional
from functools import partial

import torch
import torch.nn as nn

from torchvision.models.vision_transformer import Encoder


class SpectraTransformer(nn.Module):
    def __init__(
            self,
            spectra_length: int = 2150,
            patch_size: int = 10,
            num_layers: int = 12,
            num_heads: int = 12,
            hidden_dim: int = 288,
            mlp_dim: int = 40,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            num_output: int = 1,
            representation_size: Optional[int] = None,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super(SpectraTransformer, self).__init__()
        torch._assert(spectra_length % patch_size == 0, "spectra_length must be divisible by patch_size!")
        self.spectra_length = spectra_length
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_output = num_output
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        self.conv_proj = nn.Conv1d(
            in_channels=1, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
        )

        seq_length = spectra_length // patch_size

        # Define positional embedding parameters
        position = torch.arange(0, seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * -(math.log(10000.0) / hidden_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, seq_length, hidden_dim), requires_grad=False)
        self.position_embedding[:, :, 0::2] = torch.sin(position * div_term)
        self.position_embedding[:, :, 1::2] = torch.cos(position * div_term)

        self.global_token = nn.Parameter(torch.zeros(1, 1, hidden_dim), requires_grad=False)
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )

        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_output)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_output)

        self.heads = nn.Sequential(heads_layers)

        # if isinstance(self.heads.head, nn.Linear):
        #     nn.init.zeros_(self.heads.head.weight)
        #     nn.init.zeros_(self.heads.head.bias)

    def process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, d, l = x.shape
        p = self.patch_size
        assert l == self.spectra_length, f"Wrong spectra length! Expected {self.spectra_length} but got {l}!"
        n_l = l // p

        # (n, d, l) -> (n, hidden_dim, n_l)
        x = self.conv_proj(x)

        # (n, hidden_dim, n_l) -> (n, n_l, hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the embedding dimension
        x = x.permute(0, 2, 1)

        x = x + self.position_embedding

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self.process_input(x)
        n = x.shape[0]

        # Expand the global token to the full batch
        batch_global_token = self.global_token.expand(n, -1, -1)
        x = torch.cat((batch_global_token, x), dim=1)

        x = self.encoder(x)
        x = x[:, 0]

        x = self.heads(x)

        return x


if __name__ == '__main__':
    batch_size = 17
    spec_len = 2150
    device = torch.device("cuda:0")

    input_tensor = torch.rand(batch_size, 1, spec_len).to(device)

    model = SpectraTransformer().to(device)
    output_tensor = model(input_tensor)
    print('output_tensor:', output_tensor.shape)
