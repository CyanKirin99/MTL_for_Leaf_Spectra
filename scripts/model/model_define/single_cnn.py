import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from collections import namedtuple



BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])

# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

DEFAULT_BLOCKS_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
              expand_ratio=1, id_skip=True, strides=1, se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
              expand_ratio=6, id_skip=True, strides=2, se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
              expand_ratio=6, id_skip=True, strides=2, se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
              expand_ratio=6, id_skip=True, strides=2, se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
              expand_ratio=6, id_skip=True, strides=1, se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
              expand_ratio=6, id_skip=True, strides=2, se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
              expand_ratio=6, id_skip=True, strides=1, se_ratio=0.25)
]

# PyTorch uses kaiming_normal_ or kaiming_uniform_ for initialization
def init_weights(m):
    if type(m) == nn.Conv1d or type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)


class SEBlock(nn.Module):
    def __init__(self, input_filters, se_ratio):
        super(SEBlock, self).__init__()
        self.num_reduced_filters = max(1, int(input_filters * se_ratio))
        self.fc1 = nn.Conv1d(input_filters, self.num_reduced_filters, kernel_size=1)
        self.fc2 = nn.Conv1d(self.num_reduced_filters, input_filters, kernel_size=1)

    def forward(self, x):
        se_tensor = x.mean(dim=2, keepdim=True)
        se_tensor = F.relu(self.fc1(se_tensor))
        se_tensor = torch.sigmoid(self.fc2(se_tensor))
        return x * se_tensor

class MBConvBlock(nn.Module):
    def __init__(self, block_args, activation, drop_rate=None):
        super(MBConvBlock, self).__init__()
        self.block_args = block_args
        self.has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip

        # Expansion phase
        inp = block_args.input_filters
        oup = block_args.input_filters * block_args.expand_ratio
        if block_args.expand_ratio != 1:
            self._expand_conv = nn.Conv1d(inp, oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm1d(oup)
        else:
            self._expand_conv = None

        # Depthwise convolution phase
        k = block_args.kernel_size
        s = block_args.strides
        self._depthwise_conv = nn.Conv1d(
            oup, oup, kernel_size=k, stride=s, padding=k//2, groups=oup, bias=False)
        self._bn1 = nn.BatchNorm1d(oup)

        # Squeeze and Excitation phase
        if self.has_se:
            self._se_block = SEBlock(oup, block_args.se_ratio)
        else:
            self._se_block = None

        # Output phase
        final_oup = block_args.output_filters
        self._project_conv = nn.Conv1d(oup, final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm1d(final_oup)

        # Skip connection and dropout
        self._skip_add = nn.quantized.FloatFunctional()
        self._drop_connect = nn.Dropout(p=drop_rate) if drop_rate else None

    def forward(self, inputs):
        x = inputs
        if self._expand_conv:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = F.relu6(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = F.relu6(x)

        if self._se_block is not None:
            x = self._se_block(x)

        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and dropout
        if self.id_skip and self.block_args.strides == 1 and \
                self.block_args.input_filters == self.block_args.output_filters:
            if self._drop_connect:
                x = self._drop_connect(x)
            x = self._skip_add.add(x, inputs)
        return x

def round_filters(filters, width_coefficient, depth_divisor):
    """Round number of filters based on width multiplier."""

    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""

    return int(math.ceil(depth_coefficient * repeats))


class SpectraCNN(nn.Module):
    def __init__(self, width_coefficient, depth_coefficient, default_resolution, dropout_rate=0.2,
                 depth_divisor=8, blocks_args=DEFAULT_BLOCKS_ARGS, include_top=True,
                 input_shape=None, pooling=None, num_classes=20, activation=F.relu6):
        super(SpectraCNN, self).__init__()
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.dropout_rate = dropout_rate
        self.depth_divisor = depth_divisor
        self.blocks_args = blocks_args
        self.include_top = include_top
        self.input_shape = input_shape
        self.pooling = pooling
        self.num_classes = num_classes
        self.activation = activation

        # Build stem
        self.stem_conv = nn.Conv1d(1, round_filters(32, width_coefficient, depth_divisor),
                                   kernel_size=3, stride=2, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm1d(round_filters(32, width_coefficient, depth_divisor))

        # Build blocks
        self.blocks = nn.ModuleList()
        for idx, block_args in enumerate(blocks_args):
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, width_coefficient, depth_divisor),
                output_filters=round_filters(block_args.output_filters, width_coefficient, depth_divisor),
                num_repeat=round_repeats(block_args.num_repeat, depth_coefficient))
            for bidx in range(block_args.num_repeat):
                if bidx > 0:
                    block_args = block_args._replace(input_filters=block_args.output_filters, strides=1)
                self.blocks.append(MBConvBlock(block_args, activation, dropout_rate))

        # Build top
        self.top_conv = nn.Conv1d(round_filters(320, width_coefficient, depth_divisor),
                                  round_filters(320, width_coefficient, depth_divisor),
                                  kernel_size=1, bias=False)
        self.top_bn = nn.BatchNorm1d(round_filters(320, width_coefficient, depth_divisor))

        # Classifier
        if include_top:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
            self.fc = nn.Linear(round_filters(320, width_coefficient, depth_divisor), num_classes)

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.activation(x)

        for block in self.blocks:
            x = block(x)

        x = self.top_conv(x)
        x = self.top_bn(x)
        x = self.activation(x)

        if self.include_top:
            x = self.avg_pool(x)
            x = torch.flatten(x, 1)
            if self.dropout is not None:
                x = self.dropout(x)
            x = self.fc(x)

        return x

    @staticmethod
    def B0(input_shape, pooling='max', include_top=True, num_classes=20):
        return SpectraCNN(1.0, 1.0, 224, 0.2, include_top=include_top,
                            input_shape=input_shape, pooling=pooling, num_classes=num_classes)

    @staticmethod
    def B1(input_shape, pooling='max', include_top=True, num_classes=20):
        return SpectraCNN(1.0, 1.1, 240, 0.2, include_top=include_top,
                            input_shape=input_shape, pooling=pooling, num_classes=num_classes)

    @staticmethod
    def B2(input_shape, pooling='max', include_top=True, num_classes=20):
        return SpectraCNN(1.1, 1.2, 260, 0.3, include_top=include_top,
                            input_shape=input_shape, pooling=pooling, num_classes=num_classes)

    @staticmethod
    def B3(input_shape, pooling='max', include_top=True, num_classes=20):
        return SpectraCNN(1.2, 1.4, 300, 0.3, include_top=include_top,
                            input_shape=input_shape, pooling=pooling, num_classes=num_classes)

    @staticmethod
    def B4(input_shape, pooling='max', include_top=True, num_classes=20):
        return SpectraCNN(1.4, 1.8, 380, 0.4, include_top=include_top,
                            input_shape=input_shape, pooling=pooling, num_classes=num_classes)

    @staticmethod
    def B5(input_shape, pooling='max', include_top=True, num_classes=20):
        return SpectraCNN(1.6, 2.2, 456, 0.4, include_top=include_top,
                            input_shape=input_shape, pooling=pooling, num_classes=num_classes)

    @staticmethod
    def B6(input_shape, pooling='max', include_top=True, num_classes=20):
        return SpectraCNN(1.8, 2.6, 528, 0.5, include_top=include_top,
                            input_shape=input_shape, pooling=pooling, num_classes=num_classes)

    @staticmethod
    def B7(input_shape, pooling='max', include_top=True, num_classes=20):
        return SpectraCNN(2.0, 3.1, 600, 0.5, include_top=include_top,
                            input_shape=input_shape, pooling=pooling, num_classes=num_classes)

    @staticmethod
    def L2(input_shape, pooling='max', include_top=True, num_classes=20):
        return SpectraCNN(4.3, 5.3, 800, 0.5, include_top=include_top,
                            input_shape=input_shape, pooling=pooling, num_classes=num_classes)



if __name__ == '__main__':
    model = SpectraCNN.L2(input_shape=128, num_classes=10)
    inp = torch.randn(11, 1, 2150)
    out = model(inp)
    print(out.shape)