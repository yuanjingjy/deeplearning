import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers, datasets
(x,y),(x_val,y_val) = datasets.mnist.load_data()
x = 2*tf.convert_to_tensor(x,dtype=tf.float32)/255.-1#转换为张量，缩放到-1~1
y = tf.convert_to_tensor(y,dtype=tf.int32)#转换为张量
y = tf.one_hot(y,depth=10)#one-hot编码
print(x.shape,y.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((x,y))#构建数据集对象
train_dataset = train_dataset.batch(128)#批量训练
test_dataset = tf.data.Dataset.from_tensor_slices((x_val,y_val))
test_dataset = test_dataset.batch(128)

lr = 1e-3
# model = keras.Sequential([#3个非线性层的嵌套模型
#     layers.Dense(256,activation='relu'),
#     layers.Dense(128,activation='relu'),
#     layers.Dense(10)])

#创建每个非线性函数w，b参数张量
w1 = tf.Variable(tf.random.truncated_normal([784,256],stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256,128],stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128,10],stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

for epoch in range(100):
    for step, (x,y) in enumerate(train_dataset):#for every batch
        x = tf.reshape(x, (-1, 28 * 28))
        with tf.GradientTape() as tape:  # 构建梯度记录环境
            # 打平，[b,28,28] => [b,784]

            # step1.得到模型的输出output
            # [b,784] => [b,10]
            # out = model(x)

            # 完成第一个非线性函数的计算，这里显示地进行Broadcasting
            # [b,784]@[784,256]+[256] => [b,256]+[256] => [b,256]+[b,256]
            h1 = x @ w1 + tf.broadcast_to(b1, [x.shape[0], 256])
            h1 = tf.nn.relu(h1)
            # [b,256] => [b,128]
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            # [b,128] =>[b,10]
            h3 = h2 @ w3 + b3
            out = tf.nn.relu(h3)

            # 计算损失值
            loss = tf.square(y - out)
            # mean:scalar
            loss = tf.reduce_mean(loss)
        # 通过tape.gradient()函数求得网络参数到梯度信息：
        # compute gradients
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # 更新参数：w1 = w1-lr*w1_grad
        w1.assign_sub(lr * grads[0])  # 实现w1 = w1-lr*w1_grads
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        if step % 100 == 0:
            print(epoch, step, 'loss:', float(loss))


    #test/evaluation
    total_correct, total_num = 0,0
    for step, (x,y)in enumerate(test_dataset):
        x = tf.reshape(x,[-1,28*28])
        # [b, 784] => [b, 256] => [b, 128] => [b, 10]
        h1 = tf.nn.relu(x@w1+b1)
        h2 = tf.nn.relu(h1@w2+b2)
        out= h2*w3+b3

        #out :[b,10]~R
        #prob:[b,10]~[0,1]
        prob = tf.nn.softmax(out,axis=1)
        #[b,10]=>[b]
        #int64!!!
        pred = tf.argmax(prob,axis=1)
        pred = tf.cast(pred,dtype=tf.int32)
        #y:[b]
        #[b],int32
        #print(pred.dtype,y.dtype)

        correct = tf.cast(tf.equal(pred,y),dtype=tf.int32)
        correct = tf.reduce_sum(correct)

        total_correct += int(correct)
        total_num += x.shape[0]
    acc = total_correct/total_num
    print("test acc:",acc)




