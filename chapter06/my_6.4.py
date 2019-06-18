import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

#加载Iris数据集，存储花萼长度作为目标值，然后开始计算图会话
iris = datasets.load_iris()
# print('1>>>>>>>>>>>'+str(iris.data))
x_vals = np.array([x[0:3] for x in iris.data])
y_vals = np.array([x[3] for x in iris.data])
# print('2>>>>>>>>>>>'+str(x_vals))
sess = tf.Session()

#因为数据集比较小，我们设置一个种子使得返回结果可复现
seed = 2
tf.set_random_seed(seed)
np.random.seed(seed)

#为了准备数据集，我们创建一个80-20分的训练集和测试集，通过min-max缩放法正则化x特征值为0到1之间
train_indices = np.random.choice(len(x_vals),round(len(x_vals)*0.8),replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))

x_vals_train = x_vals[train_indices]
# print('3>>>>>>>>>>>'+str(x_vals_train))
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]
#通过min-max缩放法正则化x特征值为0到1之间
def normailize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min)/(col_max-col_min)

#np.nan_to_num	0代替nan
x_vals_train = np.nan_to_num(normailize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normailize_cols(x_vals_test))

# 5. 现在为数据集和目标值声明批量大小和占位符
batch_size = 50
x_data = tf.placeholder(shape=[None, 3], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 6. 这一步相当重要，声明有合适形状的模型变量。我们能声明隐藏层为任意大小，本例中设置为有五个隐藏节点
hidden_layer_nodes = 5
A1 = tf.Variable(tf.random_normal(shape=[3,hidden_layer_nodes]))
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))
A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes,1]))
b2 = tf.Variable(tf.random_normal(shape=[1]))


# 7. 分两步声明训练模型: 第一步，创建一个隐藏层输出；第二步，创建训练模型的最后输出
#在神经网络前向传播的过程中，经常可见如下两种形式的代码：
# tf.multiply（）两个矩阵中对应元素各自相乘
# tf.add(tf.matmul(x, w), b)
# tf.nn ：提供神经网络相关操作的支持，包括卷积操作（conv）、池化操作（pooling）、归一化、loss、分类操作、embedding、RNN、Evaluation。
# tf.layers：主要提供的高层的神经网络，主要和卷积相关的，tf.nn会更底层一些。
# tf.contrib：tf.contrib.layers提供够将计算图中的 网络层、正则化、摘要操作、是构建计算图的高级操作，但是tf.contrib包含不稳定和实验代码，有可能以后API会改变。

#tf.nn.relu(features, name = None) 这个函数的作用是计算激活函数 relu，即 max(features, 0)。即将矩阵中每行的非最大值置0。
hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data, A1), b1))
final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output, A2), b2))

#定义均方误差作为损失函数
#tf.reduce_mean 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值
loss = tf.reduce_mean(tf.square(y_target - final_output))

#声明优化算法，初始化模型变量
my_opt = tf.train.GradientDescentOptimizer(0.005)
train_step = my_opt.minimize(loss)
init = tf.initialize_all_variables()
sess.run(init)

#


