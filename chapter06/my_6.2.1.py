import tensorflow as tf

#创建会话
sess = tf.Session()
#声明模型变量、输入数据集和占位符。本例输入数据为5，所以乘法因子为10，可以得到50的预期值（5*10=50)
a = tf.Variable(tf.constant(1.))
x_val = 5.
x_data = tf.placeholder(dtype=tf.float32)
#增加操作到计算图中,multiply用于数值计算
multiplication = tf.multiply(a,x_data)
#声明损失函数;输出结果与预期目标值(50)之间的L2距离函数
loss = tf.square(tf.subtract(multiplication,50.))
#初始化模型变量，声明标准梯度下降优化算法
init = tf.initialize_all_variables()
sess.run(init)
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)
#优化模型输出结果，连续输入值5,反向传播损失函数来更新模型变量以达到10
print('Optimizing aMultiplication Gate Output to 50.')
for i in range(10):
    sess.run(train_step,feed_dict={x_data:x_val})
    a_val = sess.run(a)
    mult_output = sess.run(multiplication,feed_dict={x_data:x_val})
    print(str(a_val) + ' * ' + str(x_val) + ' = ' + str(mult_output))