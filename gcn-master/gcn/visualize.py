import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 加载数据
embeddings = np.load('node_embeddings.npy')  # [num_nodes, num_classes]
labels = np.load('node_labels.npy')          # [num_nodes]
added_idx = np.load('added_idx.npy')         # [n_injected_nodes]
#isolated_idx = np.load('isolated_idx.npy')   # [n_isolated_nodes]

# t-SNE降维
tsne = TSNE(n_components=2, random_state=0)
embeddings_2d = tsne.fit_transform(embeddings)

# 绘制所有节点
plt.figure(figsize=(8, 8))
num_classes = np.max(labels) + 1
for class_id in range(num_classes):
    idx = np.where(labels == class_id)[0]
    plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=f'Class {class_id}', alpha=0.5, s=20)

'''
# 高亮显示被孤立的节点
plt.scatter(
    embeddings_2d[isolated_idx, 0],
    embeddings_2d[isolated_idx, 1],
    c='black', marker='x', s=100, label='Isolated Node(s)'
)
'''


# 高亮注入节点
plt.scatter(
    embeddings_2d[added_idx, 0],
    embeddings_2d[added_idx, 1],
    c='red', marker='*', s=200, label='Injected Node(s)'
)


plt.legend()

#plt.title('t-SNE Visualization of Node Embeddings (Isolated Nodes Highlighted)')
#plt.savefig("t-SNE Visualization of Node Embeddings (Isolated Nodes Highlighted).png")

#plt.title('t-SNE Visualization of Node Embeddings (Injected Nodes Highlighted)')
#plt.savefig("t-SNE Visualization of Node Embeddings (Injected Nodes Highlighted).png")

#plt.title('t-SNE Visualization of Node Embeddings (Injected Node Highlighted-Random)')
#plt.savefig("t-SNE Visualization of Node Embeddings (Injected Node Highlighted-Random).png")

plt.title('t-SNE Visualization of Node Embeddings (Injected Node Highlighted-Max Degree)')
plt.savefig("t-SNE Visualization of Node Embeddings (Injected Node Highlighted-Max Degree).png")
