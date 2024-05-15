import torch
import torch.nn as nn


class WeightingStrategy:
    def __init__(self, model, optimizer, task_names, device):
        self.model = model
        self.optimizer = optimizer
        self.task_names = task_names
        self.device = device

    def update_weights(self, losses, epoch, step):
        raise NotImplementedError


class EW(WeightingStrategy):
    def __init__(self, task_names, device):
        super().__init__(task_names)
        self.device = device
        self.task_names = task_names

    def update_weights(self, losses, epoch, step):
        weights = {task: 1.0 for task in self.task_names}
        total_loss = sum(weights[task] * losses[task] for task in self.task_names)

        return weights, total_loss


class UW(WeightingStrategy):
    def __init__(self, task_names, device):
        super().__init__(task_names)
        self.device = device
        self.task_names = task_names

        # 初始化噪声参数
        self.loss_scale = nn.Parameter(torch.tensor([-0.5] * len(self.task_names), device=self.device))

    def update_weights(self, losses, epoch, step):
        # 计算每个任务的权重
        weights = {task: 1.0 / (2 * torch.exp(scale)) for task, scale in zip(self.task_names, self.loss_scale)}
        # 归一化权重，使得它们的和为1
        total_weight = sum(weights.values())
        weights = {task: weight / total_weight for task, weight in weights.items()}

        # 更新噪声参数
        new_loss_scale = []
        for i, task in enumerate(self.task_names):
            new_scale = (losses[task] / (2 * torch.exp(self.loss_scale[i])) + self.loss_scale[i] / 2).detach()
            new_loss_scale.append(new_scale)
        self.loss_scale = nn.Parameter(torch.tensor(new_loss_scale, device=self.device))

        return weights


class DWA(WeightingStrategy):
    def __init__(self, model, optimizer, task_names, device, T=1):
        super().__init__(model, optimizer, task_names, device)
        self.T = T
        self.K = len(task_names)

        self.last_losses = {task: 0.0 for task in self.task_names}

    def update_weights(self, losses, epoch, step):
        self.optimizer.zero_grad()

        descend_rate = {task: torch.tensor(1.0) for task in self.task_names}  # 将descend_rate转换为Tensor
        for task in self.task_names:
            if self.last_losses[task] > 0:  # 避免除以零
                descend_rate[task] = losses[task] / self.last_losses[task]
            self.last_losses[task] = losses[task].detach()
        # 使用softmax函数来归一化权重
        total = sum(torch.exp(rate / self.T) for rate in descend_rate.values())
        weights = {task: self.K * (torch.exp(rate / self.T)) / total for task, rate in descend_rate.items()}

        total_loss = sum(weights[task] * losses[task] for task in self.task_names)
        total_loss.backward()

        self.optimizer.step()


# class DB_MTL:
#     def __init__(self, beta=0.9, beta_sigma=0.5):
#         self.beta = beta
#         self.beta_sigma = beta_sigma
#         self.step = 0
#         self.grad_buffer = None
#
#     def backward(self, losses, model):
#         self.step += 1
#         log_losses = {tk: torch.log(losses[tk] + 1e-8) for tk in losses.keys()}
#         grads = {tk: torch.autograd.grad(log_losses[tk], model.parameters(), retain_graph=True, allow_unused=True) for tk in log_losses.keys()}
#
#         grad_norms = {}
#         for tk in grads.keys():
#             if all(g is None for g in grads[tk]):
#                 grad_norms[tk] = 0.
#             else:
#                 grad_norms[tk] = torch.norm(torch.cat([g.view(-1) for g in grads[tk] if g is not None]))
#
#         if self.grad_buffer is None:
#             self.grad_buffer = {tk: grad_norms[tk] for tk in grad_norms.keys()}
#         else:
#             self.grad_buffer = {tk: self.beta * self.grad_buffer[tk] + (1 - self.beta) * grad_norms[tk] for tk in
#                                 grad_norms.keys()}
#
#         alpha = max(grad_norms.values())
#         weights = {tk: grad_norms[tk] / alpha for tk in grad_norms.keys()}
#
#         for tk in grads.keys():
#             for g in grads[tk]:
#                 if g is not None:
#                     g.data.mul_(weights[tk])
#
#         total_loss = sum(weights[tk] * losses[tk] for tk in losses.keys())
#
#         return weights, total_loss
# class DB_MTL(WeightingStrategy):
#     def __init__(self, model, task_names, device, beta=0.9, epsilon=1e-8):
#         super().__init__(task_names)
#         self.model = model
#         self.device = device
#         self.beta = beta
#         self.epsilon = epsilon
#         self.last_losses = {task: 0.0 for task in self.task_names}
#         self.last_grads = {task: torch.zeros_like(param) for task, param in zip(self.task_names, self.model.parameters())}
#
#     def update_weights(self, losses, epoch, step):
#         # Scale-balancing loss transformation
#         log_losses = {task: torch.log(loss + self.epsilon) for task, loss in losses.items()}
#
#         # Magnitude-balancing gradient normalization
#         self.model.zero_grad()
#         grads = {}
#         for task, loss in losses.items():
#             loss.backward(retain_graph=True)
#             grads[task] = [param.grad.clone() for param in self.model.parameters()]
#             self.model.zero_grad()
#
#         # Update gradients with exponential moving average (EMA)
#         for task, grad in grads.items():
#             self.last_grads[task] = [self.beta * last_grad + (1 - self.beta) * g for last_grad, g in zip(self.last_grads[task], grad)]
#
#         # Compute alpha_k
#         grad_norms = {task: torch.norm(torch.stack([torch.norm(g) for g in grad]), 2) for task, grad in self.last_grads.items()}
#         alpha_k = max(grad_norms.values())
#
#         # Normalize gradients
#         normalized_grads = {task: [g / (torch.norm(g, 2) + self.epsilon) for g in grad] for task, grad in self.last_grads.items()}
#
#         total_loss = sum(torch.exp(log_losses[task]) for task in self.task_names)
#         return log_losses, normalized_grads, total_loss

class DualBalancing(WeightingStrategy):
    def __init__(self, model, optimizer, task_names, device, beta=0.9, epsilon=1e-8):
        super().__init__(model, optimizer, task_names, device)
        self.beta = beta
        self.epsilon = epsilon
        self.avg_grads = {task: [torch.zeros_like(p) for p in self.model.parameters() if p.requires_grad] for task in self.task_names}
        self.trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]

    def update_weights(self, losses, epoch, step):
        self.optimizer.zero_grad()

        log_losses = {task: torch.log(losses[task] + self.epsilon) for task in self.task_names}
        grads = {task: torch.autograd.grad(log_losses[task], self.trainable_parameters, retain_graph=True, allow_unused=True)
                 for task in self.task_names}

        # Update the average gradients
        for task in self.task_names:
            self.avg_grads[task] = [self.beta * avg_grad + (1 - self.beta) * grad for avg_grad, grad in zip(self.avg_grads[task], grads[task])]

        # Compute the scale factor
        max_grad_norm = max(torch.norm(torch.cat([g.view(-1) for g in avg_grad])) for avg_grad in self.avg_grads.values())

        # Compute the aggregated gradient
        aggregated_grad = [sum((max_grad_norm / (torch.norm(g.view(-1)) + self.epsilon)) * g for g in avg_grads) for avg_grads in zip(*self.avg_grads.values())]

        # Assign the aggregated gradient to the model parameters and perform a gradient descent step
        for p, g in zip(self.trainable_parameters, aggregated_grad):
            p.grad = g
        self.optimizer.step()

