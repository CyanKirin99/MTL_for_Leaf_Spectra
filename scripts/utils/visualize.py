import torch
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_scatters(X, Y, title_str=''):
    # 数据处理
    if torch.is_tensor(X) and torch.is_tensor(Y):
        assert X.shape == Y.shape, f"X and Y must have the same shape, but got X:{X.shape} and Y:{Y.shape}"
        X = X.detach().cpu().numpy()
        Y = Y.detach().cpu().numpy()

    if isinstance(X, list) and isinstance(Y, list):
        assert len(X) == len(Y), f"X and Y must have the same length, but got X:{len(X)} and {len(Y)}"
        X = np.array(X)
        Y = np.array(Y)

    # 画散点
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(X, Y, alpha=0.1)

    # 计算精度
    r2 = r2_score(Y, X)
    rmse = np.sqrt(mean_squared_error(Y, X))
    mape = np.mean(np.abs((Y - X) / Y)) * 100

    # 线性回归
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
    ax.plot(X, slope * X + intercept, color='red', label='Linear regression line')

    # 注释
    plt.text(0.8, 0.4, f'points_num: {len(X)}', horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes)
    plt.text(0.8, 0.1, f'r_score: {r2:.3f}', horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes)
    plt.text(0.8, 0.2, f'RMSE: {rmse:.3f}', horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes)
    plt.text(0.8, 0.3, f'MAPE: {mape:.3f}', horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes)

    ax.set_aspect('equal', adjustable='box')
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')

    # 显示设置
    plt.legend()
    plt.title(title_str)

    plt.show()


def plot_importance(ips, wav='default', title_str=''):
    if wav == 'default':
        wav = np.arange(350, 2500, 1)

    assert len(ips) == len(wav), f'Expect same length of importance and wav, got ips:{len(ips)}|wav:{len(wav)}'

    # 画散点
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(wav, ips, color='steelblue', linewidth=2)
    ax.plot(wav, np.zeros(len(wav)), linestyle='--', color='black', linewidth=1)

    ax.set_xlabel('Wavelength(nm)')
    ax.set_ylabel('Importance')

    plt.title(title_str)
    plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    x = np.random.rand(100)
    y = 2 * x + np.random.normal(0, 0.1, 100)
    plot_scatters(2 * x, y)
