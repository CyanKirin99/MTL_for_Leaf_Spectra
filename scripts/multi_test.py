import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from model.model_define.mtl_transformer import MTLTransformer
from data.dataset_define import MultiTraitsDataset
from utils.visualize import plot_scatters
from utils.loss import compute_loss_sparse, compute_loss


trait_names = ['CHLab', 'CAR', 'EWT', 'LMA']
model_dir = 'model/model_param/Multi/model_checkpoint_200.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 50

# 加载测试数据
with open('data/train_files.txt', 'r') as f:
    train_files = [line.strip() for line in f]
with open('data/test_files.txt', 'r') as f:
    test_files = [line.strip() for line in f]

train_dataset = MultiTraitsDataset(train_files, trait_names)
scalers = train_dataset.scalers
test_dataset = MultiTraitsDataset(test_files, trait_names, scalers)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 加载模型
model = MTLTransformer(task_names=trait_names, device=device).to(device)
checkpoint = torch.load(model_dir)
model.load_state_dict(checkpoint['model_state_dict'])

criterion = nn.HuberLoss()


# 测试
model.eval()
predictions = {tk: [] for tk in trait_names}
ground_truth = {tk: [] for tk in trait_names}

with torch.no_grad():
    for inputs, targets in test_dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        losses = compute_loss(criterion, outputs, targets, device)

        for tk in trait_names:
            mask = ~torch.isnan(targets[tk])
            valid_index = torch.nonzero(mask).squeeze()

            outputs_ = outputs[tk][valid_index].squeeze()
            targets_ = targets[tk][valid_index]

            if targets[tk][valid_index].dim() == 0:  # 如果是标量
                predictions[tk].append(outputs_.item())
                ground_truth[tk].append(targets_.item())
            else:
                predictions[tk].extend(outputs_.cpu().tolist())
                ground_truth[tk].extend(targets_.cpu().tolist())
        print(f'Test Loss: {sum(losses.values()).item()}')

# 画散点图
for tk in trait_names:
    X = np.array(ground_truth[tk]) * scalers[tk]['range'] + scalers[tk]['min']
    Y = np.array(predictions[tk]) * scalers[tk]['range'] + scalers[tk]['min']
    plot_scatters(X, Y, title_str=tk)
