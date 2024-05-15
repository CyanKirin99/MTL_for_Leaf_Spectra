
def describe_param_stats(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f'NAME:{name},\t SHAPE:{param.shape},\t  MEAN:{param.data.mean().cpu().data.numpy():.5f},\t'
                  f' STD:{param.data.std().cpu().data.numpy():.5f}')


def count_param_number(model):
    num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {num}')
    return num


def estimate_param_size(model):
    total_params = count_param_number(model)
    total_bytes = total_params * 4  # 32位浮点数占用4字节
    param_size = total_bytes / (1024 ** 2)
    print(f'Parameter size(MB): {param_size:.3f}')
    return param_size  # MB
