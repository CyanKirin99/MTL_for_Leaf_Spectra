import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.utils.tensorboard import SummaryWriter

from model.model_define.mtl_transformer import MTLTransformer, SingleTransformer
from data.dataset_define import MultiTraitsDataset
from weighting import DualBalancing, DWA, EW
from utils.train_control_modual import UnFreezer
from utils.loss import compute_loss_sparse, compute_loss
from utils.compute_param import describe_param_stats, estimate_param_size


trait_names = ['CHLab', 'CAR', 'EWT', 'LMA']
log_dir = 'runs/Multi/temp'
model_dir = 'model/model_param/Multi'
# module_names = ['head', 'rep_encoder.2', 'rep_encoder.1', 'rep_encoder.0', 'embedding']
module_names = ['DoNotOpenAnything','head', 'rep_encoder.2', 'rep_encoder.1', 'rep_encoder.0']
always_train_modules = ['attention']


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 50
learning_rate = 1e-4
epoch_num = 2000

# 数据
with open('data/train_files.txt', 'r') as f:
    train_files = [line.strip() for line in f]
with open('data/val_files.txt', 'r') as f:
    val_files = [line.strip() for line in f]

train_dataset = MultiTraitsDataset(train_files, trait_names)
scalers = train_dataset.scalers
val_dataset = MultiTraitsDataset(val_files, trait_names, scalers)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


def init_model_optim(from_type, if_freeze=False, module_names=None, always_train_modules=None):
    model = MTLTransformer(trait_names, device).to(device).float()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    unfreezer = UnFreezer(model=model,
                          module_names=module_names,
                          always_train_modules=always_train_modules) if if_freeze else None

    # 用单任务模型参数初始化多任务模型
    if from_type == 'single':
        single_model = SingleTransformer(trait_names[0], device).to(device).float()
        single_model_ckp = torch.load("model/model_param/Single/just_know.pth")
        single_model.load_state_dict(single_model_ckp['model_state_dict'])
        model.embedding.load_state_dict(single_model.embedding.state_dict())
        for tk in model.task_with_share:
            for i in range(model.num_blocks):
                model.rep_encoder[tk][i].load_state_dict(single_model.rep_encoder[i].state_dict())
        for tk in model.task_names:
            model.head[tk].load_state_dict(single_model.head.state_dict())
        start_epoch = 0
        step = 0

    elif from_type == 'multi':
        mtl_model_ckp = torch.load(f"{model_dir}/pretrain_classic.pth")
        try:
            model.load_state_dict(mtl_model_ckp['model_state_dict'])
            model.load_state_dict(mtl_model_ckp['model_state_dict'])
            optimizer.load_state_dict(mtl_model_ckp['optimizer_state_dict'])
            start_epoch = mtl_model_ckp['epoch']
            step = mtl_model_ckp['step']
            print(f"Loaded checkpoint from epoch-{start_epoch}, step-{step}")
        except RuntimeError as e:
            print("failed to load model from checkpoint", e)

    return model, optimizer, unfreezer, start_epoch, step


model, optimizer, unfreezer, start_epoch, step = init_model_optim(from_type='multi',
                                                                  if_freeze=True,
                                                                  module_names=module_names,
                                                                  always_train_modules=always_train_modules)
weighting_strategy = DWA(model, optimizer, trait_names, device)
criterion = nn.HuberLoss()
writer = SummaryWriter(log_dir=log_dir)


for epoch in range(start_epoch, epoch_num):
    model.train()
    unfreezer.step(epoch - start_epoch)

    for i, (inputs, targets) in enumerate(train_dataloader):
        step += 1
        inputs = inputs.to(device)
        outputs = model(inputs)
        losses = compute_loss(criterion, outputs, targets, device)

        weighting_strategy.update_weights(losses, epoch, step)

        for tk in trait_names:
            writer.add_scalar(f'Loss/train/{tk}', losses[tk].item(), step)
            print(
                f"{tk}|\tEpoch [{epoch + 1}/{epoch_num}]|Step [{i + 1}/{len(train_dataloader)}]|Loss: {losses[tk].item():.6f}")
        print('\n')

    with torch.no_grad():
        model.eval()
        val_loss = {tk: 0. for tk in trait_names}
        for inputs, targets in val_dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            losses = compute_loss(criterion, outputs, targets, device)
            val_loss = {tk: val_loss[tk] + losses[tk].item() for tk in losses.keys()}

        for tk in trait_names:
            writer.add_scalar(f'Loss/val/{tk}', val_loss[tk], epoch + 1)
            print(f"{tk}|\tEpoch [{epoch + 1}/{epoch_num}]|Loss: {val_loss[tk]:.6f}")
        print('\n')

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step
        }, f'{model_dir}/model_checkpoint_{epoch + 1}.pth')
