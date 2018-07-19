---
layout:     post
title:      tensorboard的使用
subtitle:   Get start tensorboard
date:       2018-04-25
author:     xhhszc
catalog: true
tags:
    - tensorflow
    - tensorboard
---

# tensorboard的使用
----

# 1. 在模型中加入要可视化的元素：
以下代码摘自TensorFlow官网
有中文注释的地方即为需要在代码添加可视化的元素。
```python
def variable_summaries(var):#记录变量var的信息
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)#记录变量var的均值
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)#记录变量var的方差
    tf.summary.scalar('max', tf.reduce_max(var))#记录变量var的最大值
    tf.summary.scalar('min', tf.reduce_min(var))#记录变量var的最小值
    tf.summary.histogram('histogram', var)#记录变量var的直方图

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):  
  with tf.name_scope(layer_name):# 为变量命名，可视化时显示该名称
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights)# 记录该变量的相关信息（最大、最小、均值、直方图）
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases)# 记录该变量的相关信息（最大、最小、均值、直方图）
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.summary.histogram('pre_activations', preactivate)# 记录该变量的直方图
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)# 记录该变量的直方图
    return activations

hidden1 = nn_layer(x, 784, 500, 'layer1')

with tf.name_scope('dropout'):
  keep_prob = tf.placeholder(tf.float32)
  tf.summary.scalar('dropout_keep_probability', keep_prob)
  dropped = tf.nn.dropout(hidden1, keep_prob)

# Do not apply softmax activation yet, see below.
y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

with tf.name_scope('cross_entropy'):
  with tf.name_scope('total'):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
tf.summary.scalar('cross_entropy', cross_entropy)#记录交叉熵损失的变化

with tf.name_scope('train'):
  train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
      cross_entropy)

with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)#记录准确率的变化
```

# 2.在模型中定义好日志的路径
该路径下的日志是tensorboard可视化时所需要调用的信息。
需要注意的是，该路径设置时，可以是相对路径，但要求绝对路径中不含中文字符，否则容易出现问题。

```python
# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
merged = tf.summary.merge_all() # 定义记录所有定义元素的信息的句柄merged
train_writer = tf.summary.FileWriter('/train', sess.graph)# 定义日志文件的句柄
tf.global_variables_initializer().run()#初始化所有变量
```

# 3. 在主函数中运行对应的句柄
```python
for i in range(10000):
  if i % 10 == 0:  # 每10次就记录一次日志信息
    summary = sess.run(merged, feed_dict={x: xs, y_: ys, keep_prob: k})
    #运行merged时，需要输入与要记录的日志元素相关的信息，
    #一般情况下，需要提供所有定义的placehoder信息
    test_writer.add_summary(summary, i)#写入日志
```

# 4. 运行程序

# 5.在主函数程序路径下，启用tensorboard
```
tensorboard --logdir=your/log/direction --port=6006
```
- logdir：为第2步中定义的路径，但此处的logdir必须写绝对路径，且路径中不含中文字符。
- port： 端口号，不填的时候默认为6006，但有时候该端口可能会被占用，因此可以用这个参数换另外的端口
**想重点提醒的是，如果不使用tensorboard服务的时候，请务必使用ctrl+c关闭该进程，如果使用ctrl+z的话并不能释放掉占用的端口，下一次再重启tensorboard的话，会显示6006端口已被占用。**
#6.将服务器的6006端口重定向到自己机器上来
如果你运行的程序是在本地机器上，那么请忽略这个步骤，直接执行步骤7
但如果你运行的程序在服务器上，需要通过自己的本地电脑查看tensorboard的可视化，那么就需要执行该步骤。
```
 ssh -L 16005:127.0.0.1:6006 yourUserNameForServer@serverIP
```
- 16005为转接到本地的端口，可以自行设置
- 127.0.0.1表示本机
如果你是window系统，推荐你安装mobaxterm，因为并不是所有的shell软件都可以支持这个指令，例如我之前用的Xshell就不支持。

# 7.在浏览器中查看tensorboard
打开你的浏览器，输入地址。
如果你的程序是在本地机器上，那么输入地址为：localhost:6006
如果你的程序是在服务器上，那么输入地址为：localhost：16005
然后你就可以查看 TensorBoard 了，您会看到右上角的导航标签。每个标签代表一组可供可视化的序列化数据。
![这里写图片描述](https://img-blog.csdn.net/20180425181924902?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3hoaHN6Yw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

# Tips
步骤4、5、6的顺序并不是固定的，可以有两种顺序：4、5、6或6、4、5