import tensorflow as tf
import numpy as np

sess = tf.Session()
data_size = 25
data_1d = np.random.normal(size=data_size)
x_input_1d = tf.placeholder(dtype=tf.float32,shape=[data_size])

#定义一个卷积层的函数，接着声明一个随机过滤层，创建一个卷积层。
#许多Tensorflow的层函数是为四维数据设计的（dD=[batch size,width,height,channels）。我们
#需要调整输入数据和输出数据，包括扩展维度和降维。在本例中，批量大小为1，
#宽度为1，高度为25，颜色通道为1.为了扩展维度，使用expand_dims()函数；降维使用squeeze()
#函数。卷积层的输出结果的维度公式为output_size=(W-F+2P)/S+1，其中W为输入数据维度，F为过滤层大小，
#P是padding大小，S是步长大小
def conv_layer_1d(input_1d,my_filter):
    input_2d = tf.expand_dims(input_1d,0)
    input_3d = tf.expand_dims(input_2d,0)
    input_4d = tf.expand_dims(input_3d,3)
    convolution_output = tf.nn.conv2d(input_4d,filter=my_filter,strides=[1,1,1,1],padding="VALID")

    conv_output_1d = tf.squeeze(convolution_output)
    return conv_output_1d


my_filter = tf.Variable(tf.random_normal(shape=[1,5,1,1]))
my_convolution_output = conv_layer_1d(x_input_1d,my_filter)

#Tensorflow的激励函数默认是逐个元素进行操作，这意味着，在部分层中使用激励函数，下面创建一个激励函数并初始化
def activation(input_1d):
    return(tf.nn.relu(input_1d))

my_activation_output = activation(my_convolution_output)


