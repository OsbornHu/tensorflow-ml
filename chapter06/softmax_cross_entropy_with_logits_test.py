import tensorflow as tf
import numpy as np

# labels：和logits具有相同type和shape的张量(tensor)，,是一个有效的概率，sum(labels)=1, one_hot=True(向量中只有一个值为1.0，其他值为0.0)。
# 计算方式：对输入的logits先通过softmax函数计算，再计算它们的交叉熵，但是它对交叉熵的计算方式进行了优化，使得结果不至于溢出。
# 适用：每个类别相互独立且排斥的情况，一幅图只能属于一类，而不能同时包含一条狗和一只大象。
# output：不是一个数，而是一个batch中每个样本的loss，所以一般配合tf.reduce_mean(loss)使用。
#
# 计算公式：https://upload-images.jianshu.io/upload_images/5877934-5261229762c60598.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/717/format/webp

def softmax(x):
    sum_raw = np.sum(np.exp(x),axis=-1)
    x1 = np.ones(np.shape(x))
    for i in range(np.shape(x)[0]):
        x1[i] = np.exp(x[i])/sum_raw[i]
    return x1

y = np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,0]])# 每一行只有一个1
logits =np.array([[12,3,2],[3,10,1],[1,2,5],[4,6.5,1.2],[3,6,1]])
# 按计算公式计算
y_pred =softmax(logits)
E1 = -np.sum(y*np.log(y_pred),-1)
print(E1)
# 按封装方法计算
sess = tf.Session()
y = np.array(y).astype(np.float64)
E2 = sess.run(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits))
print(E2)

if E1.all() == E2.all():
    print("True")
else:
    print("False")
# 输出的E1，E2结果相同