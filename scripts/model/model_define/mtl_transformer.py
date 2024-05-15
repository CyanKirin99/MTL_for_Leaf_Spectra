import math
import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Any, Callable, Dict, List, NamedTuple, Optional
from functools import partial
from torchvision.models.vision_transformer import Encoder, MLPBlock

from scripts.utils.compute_param import describe_param_stats, estimate_param_size


class Embedding(nn.Module):
    def __init__(self,
                 spectra_length: int = 2150,
                 patch_size: int = 10,
                 hidden_dim: int = 288,
                 ):
        super(Embedding, self).__init__()
        assert spectra_length % patch_size == 0, "spectra_length must be divisible by patch_size!"
        self.spectra_length = spectra_length

        self.conv_proj = nn.Conv1d(
            in_channels=1, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
        )

        seq_length = spectra_length // patch_size
        position = torch.arange(0, seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * -(math.log(10000.0) / hidden_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, seq_length, hidden_dim), requires_grad=False)
        self.position_embedding[:, :, 0::2] = torch.sin(position * div_term)
        self.position_embedding[:, :, 1::2] = torch.cos(position * div_term)

    def forward(self, x: torch.Tensor):
        n, d, l = x.shape
        assert l == self.spectra_length, f"Wrong spectra length! Expected {self.spectra_length} but got {l}!"

        x = self.conv_proj(x)
        x = x.permute(0, 2, 1)

        x = x + self.position_embedding

        return x

#
# class EncoderLike(Encoder):
#     def __init__(self, *args, **kwargs):
#         super(EncoderLike, self).__init__(*args, **kwargs)
#
#     def forward(self, input: torch.Tensor):
#         assert input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}"
#         x = self.dropout(input)
#         # layers have their own residual connect
#         x = self.layers(x)
#         x = self.ln(x)
#         return x


class RepEncoder(nn.Module):
    def __init__(
            self,
            spectra_length: int = 2150,
            patch_size: int = 10,
            num_layers: int = 12,
            num_heads: int = 12,
            hidden_dim: int = 288,
            mlp_dim: int = 1152,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6), ):
        super(RepEncoder, self).__init__()

        assert spectra_length % patch_size == 0, "spectra_length must be divisible by patch_size!"
        seq_length = spectra_length // patch_size + 1
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim

        self.encoder = Encoder(seq_length, num_layers, num_heads, hidden_dim, mlp_dim,
                               dropout, attention_dropout, norm_layer)

    def forward(self, x: torch.Tensor, just_global_token=False):
        sample_shape = x.shape[1:]
        expert_shape = (self.seq_length, self.hidden_dim)
        assert sample_shape == expert_shape, f"Expert shape {expert_shape} but got {sample_shape}"

        x = self.encoder(x)

        return x[:, 0] if just_global_token else x


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim=288, num_heads=12, dropout=0.0):
        super(CrossAttention, self).__init__()

        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.mlp = MLPBlock(in_dim=hidden_dim, mlp_dim=hidden_dim * 4, dropout=dropout)

    def forward(self, input: dict, tk: str):
        # Multi-head attention
        tasks_rep_dict = {k: v for k, v in input.items() if k != "share"}
        tasks_tensor = torch.cat(list(tasks_rep_dict.values()), dim=1)
        share_rep = input['share']

        if tk == 'share':  # q from share(self), k, v from merged tasks
            attn, _ = self.attention(share_rep, tasks_tensor, tasks_tensor)
        else:  # q from self, k, v from shared
            attn, _ = self.attention(input[tk], share_rep, share_rep)
        x = input[tk] + attn

        y = self.layer_norm(x)
        y = self.mlp(y)

        return x + y


class HeadOutput(nn.Module):
    def __init__(self, hidden_dim=288, dropout=0.1):
        super(HeadOutput, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.linear2 = nn.Linear(hidden_dim * 4, 1)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
        init.zeros_(self.linear1.bias)

        init.xavier_uniform_(self.linear2.weight)
        init.zeros_(self.linear2.bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class MTLTransformer(nn.Module):
    def __init__(self, task_names, device, num_blocks=3, hidden_dim=288):
        super(MTLTransformer, self).__init__()
        self.device = device
        self.task_names = task_names
        self.num_blocks = num_blocks
        self.task_with_share = [tk for tk in task_names] + ['share']

        self.embedding = Embedding()
        self.global_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.rep_encoder = nn.ModuleDict({tk: nn.ModuleList([RepEncoder().to(device) for _ in range(num_blocks)])
                                          for tk in self.task_with_share})
        self.attention = nn.ModuleDict({tk: nn.ModuleList([CrossAttention().to(device) for _ in range(num_blocks)])
                                        for tk in self.task_with_share})
        self.head = nn.ModuleDict({tk: HeadOutput().to(device) for tk in self.task_names})

    def forward(self, inputs):
        # embedding
        emb_tensor = self.embedding(inputs)

        # expand the global_token
        n, l, h = emb_tensor.shape
        batch_global_token = self.global_token.expand(n, -1, -1)
        batch_x = torch.cat((batch_global_token, emb_tensor), dim=1)
        # clone to all tasks
        x = {tk: batch_x.clone() for tk in self.task_with_share}

        # Extraction Block: has Multi layers of task-independent rep_encoder & attention
        for i in range(self.num_blocks):
            for tk in self.task_with_share:
                x[tk] = self.rep_encoder[tk][i](x[tk])
            for tk in self.task_with_share:
                x[tk] = self.attention[tk][i](x, tk)

        # extract the global token
        x_global_token = {k: v[:, 0] for k, v in x.items()}

        # head to output
        output = {}
        for tk in self.task_names:
            output[tk] = self.head[tk](x_global_token[tk])

        return output


class SingleTransformer(nn.Module):
    def __init__(self, task_names, device, num_blocks=3, hidden_dim=288):
        super(SingleTransformer, self).__init__()
        self.device = device
        self.task_names = task_names
        self.num_blocks = num_blocks
        self.task_with_share = [tk for tk in task_names] + ['share']

        self.embedding = Embedding()
        self.global_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.rep_encoder = nn.ModuleList([RepEncoder().to(device) for _ in range(num_blocks)])
        self.head = HeadOutput()

    def forward(self, x):
        x = self.embedding(x)

        n, l, h = x.shape
        batch_global_token = self.global_token.expand(n, -1, -1)
        x = torch.cat((batch_global_token, x), dim=1)

        for encoder in self.rep_encoder:
            x = encoder(x)

        x_global_token = x[:, 0]
        y = self.head(x_global_token)

        return y


if __name__ == '__main__':
    device = torch.device("cuda:0")
    task_names = ['A', 'B', 'C', 'D']
    input_tensor = torch.randn(17, 1, 2150).to(device)

    # # 整个多任务大框架
    # model = MTLTransformer(task_names, device).to(device)
    # output = model(input_tensor)
    # for k, v in output.items():
    #     print(k, v.shape)

    # 拆出的单任务模型
    model = SingleTransformer(task_names[0], device).to(device)
    output = model(input_tensor)
    print(output.shape)

    # # 用single初始化mtl
    # model = MTLTransformer(task_names, device).to(device).float()
    # single_model = SingleTransformer(task_names[0], device).to(device).float()
    # single_model_ckp = torch.load("D:/file/Research/MTL_based_leaf_traits_estimating/scripts/"
    #                               "model/model_param/Single/just_know.pth")
    # single_model.load_state_dict(single_model_ckp['model_state_dict'])
    # model.embedding.load_state_dict(single_model.embedding.state_dict())
    # for tk in model.task_with_share:
    #     for i in range(model.num_blocks):
    #         model.rep_encoder[tk][i].load_state_dict(single_model.rep_encoder[i].state_dict())
    # for tk in model.task_names:
    #     model.head[tk].load_state_dict(single_model.head.state_dict())

    # # 描述参数
    # describe_param_stats(model)
    # param_size = estimate_param_size(model)
