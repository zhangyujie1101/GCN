from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.utils import *
from gcn.models import GCN, MLP

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

# -------- 中毒攻击：训练阶段孤立节点/注入节点 --------
'''
# 中毒攻击——随机孤立n个节点
from gcn.utils import isolate_nodes
n = 50
adj, features, isolated_idx = isolate_nodes(adj, features, n)
print("Isolated nodes:", isolated_idx)
'''

'''
# 中毒攻击——随机新增n个节点，并连接m个邻居
from gcn.utils import add_nodes
n = 50  # 新增节点数
m = 2   # 每个新节点的邻居数
adj, features, added_idx = add_nodes(adj, features, n, m, feature_mode='random')
print("Added nodes:", added_idx)

# 获取类别数
num_classes = y_train.shape[1]
num_new = len(added_idx)

# 随机分配标签
labels_new = np.zeros((num_new, num_classes))
random_classes = np.random.choice(num_classes, num_new)
labels_new[np.arange(num_new), random_classes] = 1

# 让新增节点全部参与测试集
y_train = np.vstack([y_train, np.zeros((num_new, num_classes))])
y_val = np.vstack([y_val, np.zeros((num_new, num_classes))])
y_test = np.vstack([y_test, labels_new])

train_mask = np.hstack([train_mask, np.zeros(num_new, dtype=bool)])
val_mask = np.hstack([val_mask, np.zeros(num_new, dtype=bool)])
test_mask = np.hstack([test_mask, np.ones(num_new, dtype=bool)])
'''


# 中毒攻击——注入1个节点，连接m个邻居，m为随机抽取邻居或抽取度值最大的邻居
from gcn.utils import add_nodes
n = 1  # 只注入一个节点
m = 5  # 单节点的邻居数
adj, features, added_idx = add_nodes(adj, features, n, m, feature_mode='random', neighbor_mode='max_degree')
print("Injected node:", added_idx)

# 获取类别数
num_classes = y_train.shape[1]
num_new = len(added_idx)

# 随机分配标签
labels_new = np.zeros((num_new, num_classes))
random_classes = np.random.choice(num_classes, num_new)
labels_new[np.arange(num_new), random_classes] = 1

# 让新节点参与测试集
y_train = np.vstack([y_train, np.zeros((num_new, num_classes))])
y_val = np.vstack([y_val, np.zeros((num_new, num_classes))])
y_test = np.vstack([y_test, labels_new])

train_mask = np.hstack([train_mask, np.zeros(num_new, dtype=bool)])
val_mask = np.hstack([val_mask, np.zeros(num_new, dtype=bool)])
test_mask = np.hstack([test_mask, np.ones(num_new, dtype=bool)])


# -------- 逃逸攻击：测试阶段孤立节点/注入节点 --------
'''
# 删除节点
from gcn.utils import isolate_nodes  

# 复制一份测试相关变量
adj_evasion = adj.copy()
features_evasion = features.copy()
y_test_evasion = y_test.copy()
test_mask_evasion = test_mask.copy()

# 对测试集做孤立
n = 50
adj_evasion, features_evasion, isolated_idx = isolate_nodes(adj_evasion, features_evasion, n)
print("Evasion attack: Isolated nodes:", isolated_idx)

# 特征和邻接矩阵预处理
features_evasion = preprocess_features(features_evasion)
if FLAGS.model == 'gcn':
    support_evasion = [preprocess_adj(adj_evasion)]
elif FLAGS.model == 'gcn_cheby':
    support_evasion = chebyshev_polynomials(adj_evasion, FLAGS.max_degree)
elif FLAGS.model == 'dense':
    support_evasion = [preprocess_adj(adj_evasion)]
'''

'''
# 注入节点
from gcn.utils import add_nodes

adj_evasion = adj.copy()
features_evasion = features.copy()
y_test_evasion = y_test.copy()
test_mask_evasion = test_mask.copy()

n = 50  # 注入节点数
m = 2   # 邻居数
adj_evasion, features_evasion, added_idx = add_nodes(adj, features, n, m, feature_mode='random')

# 扩展y_test和test_mask
num_classes = y_test.shape[1]
num_new = len(added_idx)
labels_new = np.zeros((num_new, num_classes))
random_classes = np.random.choice(num_classes, num_new)
labels_new[np.arange(num_new), random_classes] = 1

y_test_evasion = np.vstack([y_test_evasion, labels_new])
test_mask_evasion = np.hstack([test_mask_evasion, np.ones(num_new, dtype=bool)])

# 特征和邻接矩阵预处理
features_evasion = preprocess_features(features_evasion)
if FLAGS.model == 'gcn':
    support_evasion = [preprocess_adj(adj_evasion)]
elif FLAGS.model == 'gcn_cheby':
    support_evasion = chebyshev_polynomials(adj_evasion, FLAGS.max_degree)
elif FLAGS.model == 'dense':
    support_evasion = [preprocess_adj(adj_evasion)]
'''

'''
# 单节点注入
from gcn.utils import add_nodes

adj_evasion = adj.copy()
features_evasion = features.copy()
y_test_evasion = y_test.copy()
test_mask_evasion = test_mask.copy()

n = 1  # 注入节点数
m = 5  # 邻居数
adj_evasion, features_evasion, added_idx = add_nodes(adj_evasion, features_evasion, n, m, feature_mode='random', neighbor_mode='max_degree')

# 扩展y_test和test_mask
num_classes = y_test.shape[1]
num_new = len(added_idx)
labels_new = np.zeros((num_new, num_classes))
random_classes = np.random.choice(num_classes, num_new)
labels_new[np.arange(num_new), random_classes] = 1

y_test_evasion = np.vstack([y_test_evasion, labels_new])
test_mask_evasion = np.hstack([test_mask_evasion, np.ones(num_new, dtype=bool)])

# 特征和邻接矩阵预处理
features_evasion = preprocess_features(features_evasion)
if FLAGS.model == 'gcn':
    support_evasion = [preprocess_adj(adj_evasion)]
elif FLAGS.model == 'gcn_cheby':
    support_evasion = chebyshev_polynomials(adj_evasion, FLAGS.max_degree)
elif FLAGS.model == 'dense':
    support_evasion = [preprocess_adj(adj_evasion)]
'''

# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

'''
# 逃逸攻击：增加节点和原有节点得测试结果
test_cost, test_acc, test_duration = evaluate(features_evasion, support_evasion, y_test_evasion, test_mask_evasion, placeholders)
print("Test set (original + injected nodes) results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
'''

'''
# 逃逸攻击：删除（孤立）节点后，所有测试节点的准确率
test_cost, test_acc, test_duration = evaluate(features_evasion, support_evasion, y_test_evasion, test_mask_evasion, placeholders)
print("Test set (after node isolation, evasion attack) results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
'''

'''
feed_dict = construct_feed_dict(features, support, y_test, test_mask, placeholders)
embeddings = sess.run(model.outputs, feed_dict=feed_dict)
np.save('node_embeddings.npy', embeddings)
np.save('node_labels.npy', np.argmax(y_test, axis=1))
#np.save('added_idx.npy', added_idx)
np.save('isolated_idx.npy', isolated_idx)
'''

'''
feed_dict_evasion = construct_feed_dict(features_evasion, support_evasion, y_test_evasion, test_mask_evasion, placeholders)
embeddings_evasion = sess.run(model.outputs, feed_dict=feed_dict_evasion)
np.save('node_embeddings.npy', embeddings_evasion)
np.save('node_labels.npy', np.argmax(y_test_evasion, axis=1))
#np.save('isolated_idx.npy', isolated_idx)
np.save('added_idx.npy', added_idx)
'''