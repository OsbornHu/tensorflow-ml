import tensorflow as tf
import numpy as np

def sigmoid(x):
    return 1.0/(1+np.exp(-x))
#
# 计算方式：对输入的logits先通过sigmoid函数计算，再计算它们的交叉熵，但是它对交叉熵的计算方式进行了优化，使得的结果不至于溢出。
# 适用：每个类别相互独立但互不排斥的情况：例如一幅图可以同时包含一条狗和一只大象。
# output不是一个数，而是一个batch中每个样本的loss,所以一般配合tf.reduce_mean(loss)使用。
#计算公式图片 https://upload-images.jianshu.io/upload_images/5877934-fb698517859fea13.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/720/format/webp
# 5个样本三分类问题，且一个样本可以同时拥有多类
y = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,0]])
logits = np.array([[12,3,2],[3,10,1],[1,2,5],[4,6.5,1.2],[3,6,1]])
# 按计算公式计算
y_pred = sigmoid(logits)
E1 = -y*np.log(y_pred)-(1-y)*np.log(1-y_pred)
print(E1)     # 按计算公式计算的结果
# 按封装方法计算
sess =tf.Session()
y = np.array(y).astype(np.float64) # labels是float64的数据类型
E2 = sess.run(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=logits))
print(E2)     # 按 tf 封装方法计算

if E1.all() == E2.all():
    print("True")
else:
    print("False")
# 输出的E1，E2结果相同