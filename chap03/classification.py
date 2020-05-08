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
train_dataset = train_dataset.batch(512)#批量训练

model = keras.Sequential([#3个非线性层的嵌套模型
    layers.Dense(256,activation='relu'),
    layers.Dense(128,activation='relu'),
    layers.Dense(10)])

with tf.GradientTape() as tape:#构建梯度记录环境
    #打平，[b,28,28] => [b,784]
    x = tf.reshape(x,(-1,28*28))
    #step1.得到模型的输出output
    #[b,784] => [b,10]
    out = model(x)

#利用TensorFlow提供的自动求导函数tape.gradient(loss,model.trainable_variables)求出模型中所有的
#梯度信息，L对xita，xita{W1,b1,W2,b2,W3,b3}
grads = tape.gradient(loss, model.trainable_variables)
# w' = w - lr * grad,更新网络参数
optimizer.apply_gradients(zip(grads,model.trainable_variables))
print("test")