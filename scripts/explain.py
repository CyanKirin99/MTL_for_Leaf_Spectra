from model.model_define.mtl_transformer import *
from data.dataset_define import PretrainDataset
from torch.utils.data import DataLoader
from collections import OrderedDict
from torchvision.models.vision_transformer import EncoderBlock, Encoder


# class EncoderBlockWithAttention(EncoderBlock):
#     def forward(self, input: torch.Tensor):
#         torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
#         x = self.ln_1(input)
#         x, attn = self.self_attention(x, x, x, need_weights=True)
#         x = self.dropout(x)
#         x = x + input
#
#         y = self.ln_2(x)
#         y = self.mlp(y)
#         return x + y, attn
#
#
# class EncoderWithAttention(Encoder):
#     def __init__(self,
#                  seq_length: int,
#                  num_layers: int,
#                  num_heads: int,
#                  hidden_dim: int,
#                  mlp_dim: int,
#                  dropout: float,
#                  attention_dropout: float,
#                  norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
#                  ):
#         super().__init__(
#             seq_length,
#             num_layers,
#             num_heads,
#             hidden_dim,
#             mlp_dim,
#             dropout,
#             attention_dropout,
#             norm_layer)
#         layers: OrderedDict[str, nn.Module] = OrderedDict()
#         for i in range(num_layers):
#             layers[f"encoder_layer_{i}"] = EncoderBlockWithAttention(
#                 num_heads,
#                 hidden_dim,
#                 mlp_dim,
#                 dropout,
#                 attention_dropout,
#                 norm_layer,
#             )
#         self.layers = nn.Sequential(layers)
#
#
#     def forward(self, input: torch.Tensor):
#         torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
#         x = self.dropout(input)
#         x, attn = self.layers(x)
#         x = self.ln(x)
#         return x, attn
#
#
# class RepEncoder(RepEncoder):
#     def __init__(self,
#                  spectra_length: int = 2150,
#                  patch_size: int = 10,
#                  num_layers: int = 12,
#                  num_heads: int = 12,
#                  hidden_dim: int = 288,
#                  mlp_dim: int = 1152,
#                  dropout: float = 0.0,
#                  attention_dropout: float = 0.0,
#                  norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6)):
#         super(RepEncoder, self).__init__()
#         seq_length = spectra_length // patch_size
#         self.encoder = EncoderWithAttention(
#             seq_length,
#             num_layers,
#             num_heads,
#             hidden_dim,
#             mlp_dim,
#             dropout,
#             attention_dropout,
#             norm_layer,
#         )
#
#     def forward(self, x: torch.Tensor, just_global_token=False):
#         n, l, h = x.shape
#
#         if l != self.seq_length:
#             batch_global_token = self.global_token.expand(n, -1, -1)
#             x = torch.cat((batch_global_token, x), dim=1)
#
#         x, attn = self.encoder(x)
#
#         return x[:, 0], attn if just_global_token else x, attn
#
#
# class SingleTransformerWithAttention(SingleTransformer):
#     def __init__(self, task_names, device, num_blocks=3, hidden_dim=288):
#         super(SingleTransformerWithAttention, self).__init__(task_names, device)
#         self.rep_encoder = nn.ModuleList([RepEncoder().to(device) for _ in range(num_blocks)])
#
#     def forward(self, x):
#         x = self.embedding(x)
#
#         for encoder in self.rep_encoder:
#             x, attn = encoder(x)
#
#         x_global_token = x[:, 0]
#         y = self.head(x_global_token)
#
#         return y
#


config4transformer = {
    'dropout': 0.27,
    'batch_size': 200,
    'learning_rate': 1e-5,
    'num_epochs': 2000,
    'val_interval': 1,
    'device': torch.device('cuda:0'),
    'trait_name': '',
}

config = config4transformer

suffix = ''
config['trait_name'] = 'CHLab'
config['model_type'] = 'Single'

config['model_name'] = f"single_{config['trait_name']}{suffix}"
config['model_dir'] = f"model/model_param/{config['model_type']}/{config['model_name']}.pth"
config['log_dir'] = f"runs/{config['model_type']}/{config['model_name']}/temp"
model = SingleTransformer([config['trait_name']], config['device']).to(config['device']).float()

checkpoint = torch.load(config['model_dir'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with open('data/train_files.txt', 'r') as f:
    train_files = [line.strip() for line in f]
with open('data/test_files.txt', 'r') as f:
    test_files = [line.strip() for line in f]

train_dataset = PretrainDataset(train_files, config['trait_name'])
scalers = train_dataset.scalers
test_dataset = PretrainDataset(test_files, config['trait_name'], scalers)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

data = next(iter(test_dataloader))
inputs, labels = data
inputs, labels = inputs.to(config['device']), labels.to(config['device'])


import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention_weights(attention_weights, layer_idx=0, head_idx=0):
    """
    可视化注意力权重。

    Args:
        attention_weights: 自注意力权重 (list of tensors)
        layer_idx: 要可视化的层索引
        head_idx: 要可视化的头索引
    """
    attn = attention_weights[layer_idx][0, head_idx]  # 获取指定层和头的注意力权重
    sns.heatmap(attn.numpy(), cmap="viridis")
    plt.title(f'Attention Weights - Layer {layer_idx+1}, Head {head_idx+1}')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.show()



output = model(inputs, capture_attention=True)
# 提取注意力权重
attention_weights = single_model.get_attention_weights()
# 可视化第一个层和第一个头的注意力权重
plot_attention_weights(attention_weights, layer_idx=0, head_idx=0)
