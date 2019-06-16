import tensorflow as tf
#tf.reset_default_graph函数用于清除默认图形堆栈并重置全局默认图形
from tensorflow.python.framework import  ops
ops.reset_default_graph

#创建会话
sess = tf.Session()
#随机给变量赋一个常量初始值
a = tf.Variable(tf.constant(1.))
b = tf.Variable(tf.constant(1.))

x_val = 5.
x_data = tf.placeholder(dtype=tf.float32)

# f(x) = a * x  + b
two_gate = tf.add(tf.multiply(a,x_data),b)
#损失函数
loss = tf.square(tf.subtract(two_gate,50.))
#梯度下降优化
my_opt = tf.train.GradientDescentOptimizer(0.01)

train_step = my_opt.minimize(loss)
#初始化模型变量，声明标准梯度下降优化算法
init = tf.initialize_all_variables()
sess.run(init)

#循环训练10次
print('Optimizing aMultiplication Gate Output to 50.')
for i in range(10):
    sess.run(train_step,feed_dict={x_data:x_val})  #运行优化器
    a_val,b_val = (sess.run(a),sess.run(b)) #输入初始变量
    print('a:' + str(a_val) + 'b:' + str(b_val))
    two_gate_output = sess.run(two_gate,feed_dict={x_data:x_val})  #运行函数
    print(str(a_val) + ' * ' + str(x_val) + ' + ' + str(b_val) + ' = ' + str(two_gate_output))


