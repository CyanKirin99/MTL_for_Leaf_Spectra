import torch


def compute_loss_sparse(criterion, outputs, targets, device):
    losses = {}
    for i, tk in enumerate(outputs.keys()):
        mask = ~torch.isnan(targets[tk])
        valid_len_tk = torch.sum(mask)
        if valid_len_tk > 0:
            loss_tk = criterion(outputs[tk][mask].squeeze().to(device).float(),
                                targets[tk][mask].to(device).float())
            losses[tk] = loss_tk / valid_len_tk
        else:
            losses[tk] = torch.tensor(0., requires_grad=True).float()
    return losses


def compute_loss(criterion, outputs, targets, device):
    losses = {}
    for i, tk in enumerate(outputs.keys()):
        loss_tk = criterion(outputs[tk].squeeze().to(device).float(),
                            targets[tk].to(device).float())
        losses[tk] = loss_tk
    return losses
