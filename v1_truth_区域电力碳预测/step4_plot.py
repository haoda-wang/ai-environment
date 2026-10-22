import os
import numpy as np
import matplotlib.pyplot as plt
from step1_read_data import load_raw_data, clean_reshape
from step2_feature import make_features
from step3_model import run_all
plt.rcParams['font.sans-serif'] = ['SimHei']      # 中文字体，不加会变方框
plt.rcParams['axes.unicode_minus'] = False


def plot_pred(y_test, preds, path='fig/pred_compare.png'):
    """真实 vs 各模型预测：指标看不出的事情都在这张图上"""
    os.makedirs('fig', exist_ok=True) #

    fig, ax = plt.subplots(figsize=(11, 5))
    x = y_test.index.to_numpy()                       # 必须用日期数组当 x，防 y 轴污染
    ax.plot(x, y_test.to_numpy(), 'k-', lw=2.5, label='真实值')
    for name, p in preds.items():
        ax.plot(x, np.asarray(p), '--', lw=1.4, label=name)
    ax.set_title('2025-01 ~ 2026-09 月度火电发电量预测对比（亿千瓦时）')
    ax.set_ylabel('火电发电量（亿千瓦时）')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate() # 自动旋转 x 轴标签
    fig.tight_layout()  # 自动调整子图参数
    fig.savefig(path, dpi=150)                        # 必须在 show 之前
    plt.show()

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = make_features(clean_reshape(load_raw_data('data/月度数据.csv')))
    df,preds = run_all(x_train, y_train, x_test, y_test)
    # print(df)
    plot_pred(y_test, preds)