import matplotlib.pyplot as plt

# ============================
# 实验结果
# ============================

learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
accuracy_lr = [0.648, 0.718, 0.713, 0.702, 0.696]

hidden_units = [8, 16, 32, 64, 128]
accuracy_hidden = [0.703, 0.713, 0.714, 0.71, 0.705]

num_layers = [1, 2, 3, 4]
accuracy_layers = [0.627, 0.713, 0.678, 0.649]

dropouts = [0.0, 0.3, 0.5, 0.7]
accuracy_dropout = [0.71, 0.713, 0.713, 0.72]

weight_decays = [0, 5e-5, 5e-4, 5e-3]
accuracy_weight_decay = [0.681, 0.701, 0.713, 0.678]


# ============================
# 辅助函数：绘制子图并标注最佳点
# ============================
def plot_with_best(ax, x, y, xlabel, ylabel, title, logx=False):
    ax.plot(x, y, marker='o')
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)

    # 找到最佳点
    best_idx = max(range(len(y)), key=lambda i: y[i])
    best_x, best_y = x[best_idx], y[best_idx]

    # 标红最佳点
    ax.plot(best_x, best_y, 'ro')
    ax.annotate(f"{best_y:.2f}", (best_x, best_y),
                textcoords="offset points", xytext=(0, 10),
                ha='center', color="red", fontsize=10, fontweight="bold")


# ============================
# 绘制 2x3 子图
# ============================
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
axs = axs.ravel()

plot_with_best(axs[0], learning_rates, accuracy_lr,
               "Learning Rate", "Accuracy", "Accuracy vs Learning Rate", logx=True)

plot_with_best(axs[1], hidden_units, accuracy_hidden,
               "Hidden Units", "Accuracy", "Accuracy vs Hidden Units")

plot_with_best(axs[2], num_layers, accuracy_layers,
               "Number of Layers", "Accuracy", "Accuracy vs Number of Layers")

plot_with_best(axs[3], dropouts, accuracy_dropout,
               "Dropout Rate", "Accuracy", "Accuracy vs Dropout Rate")

plot_with_best(axs[4], weight_decays, accuracy_weight_decay,
               "Weight Decay (L2)", "Accuracy", "Accuracy vs Weight Decay", logx=True)

# 关掉最后一个空子图
axs[5].axis("off")

# 调整布局
plt.tight_layout()
plt.savefig("hyperparameter_tuning_results.png")
# plt.show()
plt.close()