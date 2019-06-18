import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.Session()
#使用Tensorflow和Numpy模块的随机数生成器，对于相同的随机种子集，我们应该能够复现
tf.set_random_seed(5)
np.random.seed(42)
#声明批量大小、模型变量、数据集和占位符。在计算图中为两个相似的神经网络模型（权激励函数不同）传入正态分布数据
batch_size = 5
a1 = tf.Variable(tf.random_normal(shape=[1,1]))
b1 = tf.Variable(tf.random_uniform(shape=[1,1]))
a2 = tf.Variable(tf.random_normal(shape=[1,1]))
b2 = tf.Variable(tf.random_uniform(shape=[1,1]))

#参数的意义为：

#loc：float
 #   此概率分布的均值（对应着整个分布的中心centre）
#scale：float
 #   此概率分布的标准差（对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高）
#size：int or tuple of ints
#    输出的shape，默认为None，只输出一个值
#我们更经常会用到的np.random.randn(size)所谓标准正态分布（μ=0,σ=1μ=0,σ=1），对应于np.random.normal(loc=0, scale=1, size)
x = np.random.normal(2,0.1,500)
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)

#声明两个训练模型，即sigmoid激励模型和ReLU激励模型
sigmoid_activation = tf.sigmoid(tf.add(tf.matmul(x_data, a1), b1))
relu_activation = tf.nn.relu(tf.add(tf.matmul(x_data, a2), b2))

# 4. 损失函数都采用模型输出和预期值0.75之间的差值的L2范数平均
loss1 = tf.reduce_mean(tf.square(tf.subtract(sigmoid_activation, 0.75)))
loss2 = tf.reduce_mean(tf.square(tf.subtract(relu_activation, 0.75)))

#梯度下降优化
my_opt = tf.train.GradientDescentOptimizer(0.01)

#训练直到损失函数最小
train_step_sigmoid = my_opt.minimize(loss1)
train_step_relu= my_opt.minimize(loss2)

init = tf.initialize_all_variables()
sess.run(init)
#遍历迭代训练模型，每个模型迭代750次，保存损失函数输出和激励函数的返回值，以便后续绘图
loss_vec_sigmoid = []
loss_vec_relu = []
activation_sigmoid = []
activation_relu = []

for i in range(750):
    # np.random.choice(a=5, size=3, replace=False, p=None)参数意思分别 是从a 中以概率P，随机选择3个
    #print('len(x):' + str(len(x)))
    rand_indices = np.random.choice(len(x) ,size=batch_size)
    #矩阵转置
    x_vals = np.transpose([x[rand_indices]])
    print('x_vals:' + str(x_vals))
    sess.run(train_step_sigmoid, feed_dict={x_data: x_vals})
    sess.run(train_step_relu, feed_dict={x_data: x_vals})
    #保存损失函数输出和激励函数的返回值，以便后续绘图
    loss_vec_sigmoid.append(sess.run(loss1, feed_dict={x_data: x_vals}))
    loss_vec_relu.append(sess.run(loss2, feed_dict={x_data: x_vals}))

    activation_sigmoid.append(np.mean(sess.run(sigmoid_activation, feed_dict={x_data: x_vals})))
    activation_relu.append(np.mean(sess.run(relu_activation, feed_dict={x_data: x_vals})))


# 7. 下面是绘制损失函数和激励函数的代码
# 基于ReLU激励函数的形式，它将比sigmoid激励函数返回更多的0值，我们认为该行为是一种稀疏的，稀疏性导致收敛速度加快，但是损失了一部分梯度控制能力。
#相反，sigmoid激励函数具有良好的梯度控制，不会出现ReLU激励函数那样的极值。
plt.plot(activation_sigmoid, 'k-', label='Sigmoid Activation')
plt.plot(activation_relu, 'r--', label='Relu Activation')
plt.ylim([0, 1.0])
plt.title('Activation Outputs')
plt.xlabel('Generation')
plt.ylabel('Outputs')
plt.legend(loc='upper right')
plt.show()
plt.plot(loss_vec_sigmoid, 'k-', label='Sigmoid Loss')
plt.plot(loss_vec_relu, 'r--', label='Relu Loss')
plt.ylim([0, 1.0])
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
