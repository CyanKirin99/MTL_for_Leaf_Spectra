import numpy as np
import time
import glob
import shap
from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.model_define.mtl_transformer import SingleTransformer
from data.dataset_define import PretrainDataset
from utils.visualize import plot_scatters, plot_importance
from utils.train_control_modual import UnFreezer


config4transformer = {
    'batch_size': 50,
    'learning_rate': 5e-5,
    'num_epochs': 2000,
    'val_interval': 1,
    'device': torch.device('cuda:0'),
    'trait_name': '',
}

config = config4transformer

# 模型
suffix = ''
config['trait_name'] = 'CHLab'
config['model_type'] = 'Single'

config['model_name'] = f"single_{config['trait_name']}{suffix}"
config['model_dir'] = f"model/model_param/{config['model_type']}/{config['model_name']}.pth"
config['log_dir'] = f"runs/{config['model_type']}/{config['model_name']}/temp"

model = SingleTransformer([config['trait_name']], config['device']).to(config['device'])

config['model'] = model
config['model_type'] = config['model_type']


def train_main(config, gradual_unfreeze=False, module_names=None):
    print('Start Training')
    device = config['device']
    model = config['model'].to(device).float()
    trait_name = config['trait_name']

    model_dir = config['model_dir']
    log_dir = config['log_dir']

    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    val_interval = config['val_interval']

    # 数据
    with open('data/train_files.txt', 'r') as f:
        train_files = [line.strip() for line in f]
    with open('data/val_files.txt', 'r') as f:
        val_files = [line.strip() for line in f]

    train_dataset = PretrainDataset(train_files, trait_name)
    scalers = train_dataset.scalers
    val_dataset = PretrainDataset(val_files, trait_name, scalers)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 训练设置
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    writer = SummaryWriter(log_dir=log_dir)

    try:
        checkpoints = glob.glob(model_dir)
        if checkpoints:
            checkpoint_file = checkpoints[-1]
            checkpoint = torch.load(checkpoint_file)

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            step = checkpoint['step']
            print(f"Loaded checkpoint from epoch-{start_epoch}, step-{step}")
        else:
            print("No checkpoint found, training from scratch.")
            start_epoch = 0
            step = 0
    except FileNotFoundError:
        print("No checkpoint found, training from scratch.")
        start_epoch = 0
        step = 0

    if gradual_unfreeze:
        unfreezer = UnFreezer(model, module_names)

    # 训练
    start_time = time.time()
    best_val_loss = torch.inf
    for epoch in range(start_epoch, num_epochs):
        model.train()
        # 如果启用了逐步解冻，我们在每个epoch开始时解冻一个层
        if gradual_unfreeze:
            unfreezer.step(epoch - start_epoch)

        for i, data in enumerate(train_dataloader):
            step += 1
            inputs, labels = data
            inputs, labels = inputs.float().to(device), labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            print(f'Epoch: {epoch + 1}| iter: {i + 1}/{len(train_dataloader)}| Iter_Loss: {loss.item():.5f}')
            writer.add_scalar(tag='loss/train', scalar_value=loss.item(), global_step=step)

        # 验证
        if epoch % val_interval == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for i, data in enumerate(val_dataloader):
                    inputs, labels = data
                    inputs, labels = inputs.float().to(device), labels.float().to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), labels)
                    val_loss += loss.item()

                print(f'Validation|Epoch: {epoch + 1} | Val_Loss: {val_loss:.5f}')
                writer.add_scalar(tag='loss/val', scalar_value=val_loss, global_step=epoch + 1)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'step': step
                    }, model_dir)

    # 保存最后的参数文件
    torch.save(model.state_dict(), f'{model_dir.split(".pth")[0]}_last.pth')
    writer.close()

    end_time = time.time()
    print(f'Training time: {end_time - start_time:.1f}')


def test_main(config, if_explain=False):
    print('Start Testing')
    device = config['device']
    model = config['model'].to(device).float()
    model_dir = config['model_dir']
    trait_name = config['trait_name']
    batch_size = config['batch_size']

    try:
        checkpoints = glob.glob(model_dir)
        if checkpoints:
            checkpoint_file = checkpoints[-1]
            checkpoint = torch.load(checkpoint_file)

            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {checkpoint_file.split('/')[-1]}")
        else:
            print("No checkpoint found")
    except FileNotFoundError:
        print("No checkpoint found, training from scratch.")

    model.eval()
    # 数据
    with open('data/train_files.txt', 'r') as f:
        train_files = [line.strip() for line in f]
    with open('data/test_files.txt', 'r') as f:
        test_files = [line.strip() for line in f]

    train_dataset = PretrainDataset(train_files, trait_name)
    scalers = train_dataset.scalers
    test_dataset = PretrainDataset(test_files, trait_name, scalers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 测试
    all_inputs, all_outputs, all_labels = [], [], []
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            all_inputs.append(inputs)
            all_outputs.append(outputs.squeeze().detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    # all_inputs = torch.concatenate(all_inputs, dim=0)
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    all_outputs = all_outputs * scalers['range'] + scalers['min']
    all_labels = all_labels * scalers['range'] + scalers['min']

    plot_scatters(all_labels, all_outputs, title_str=trait_name)

    if if_explain:
        explainer = shap.DeepExplainer(model, inputs)
        sample = inputs[-2:-1]
        shap_values = explainer.shap_values(sample, check_additivity=False)

        plot_importance(shap_values.squeeze(), title_str=trait_name)


if __name__ == "__main__":
    module_names = ['head', 'rep_encoder.2', 'rep_encoder.1', 'rep_encoder.0', 'embedding']
    # train_main(config)
    # train_main(config, True, module_names)
    test_main(config)
