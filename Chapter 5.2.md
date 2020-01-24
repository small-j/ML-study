## Chapter 5-2


```python
import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()

data = np.loadtxt('./data.csv', delimiter=',',
                 unpack=True, dtype='float32')

x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])

### 신경망 모델 구성
global_step = tf.Variable(0, trainable=False, name='global_step')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
    L1 = tf.nn.relu(tf.matmul(X, W1))

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.))
    L2 = tf.nn.relu(tf.matmul(L1, W2))

with tf.name_scope('output'):
    W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.))
    model = tf.matmul(L2, W3)

with tf.name_scope('optimizer'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(cost, global_step=global_step)
    
    tf.summary.scalar('cost',cost)
    
### 신경망 모델 학습
sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())

ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())
    
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs', sess.graph)

for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    print('Step %d, '% sess.run(global_step),
     'Cost: %.3f' % sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    summary = sess.run(merged, feed_dict={X: x_data, Y: y_data})
    writer.add_summary(summary, global_step=sess.run(global_step))
    
saver.save(sess, './model/dnn.ckpt', global_step=global_step)

###결과확인
prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
print('실제값:', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
```

    WARNING:tensorflow:From c:\users\rlawl\appdata\local\programs\python\python37\lib\site-packages\tensorflow_core\python\compat\v2_compat.py:65: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
    Instructions for updating:
    non-resource variables are not supported in the long term
    Step 1,  Cost: 0.914
    Step 2,  Cost: 0.852
    Step 3,  Cost: 0.799
    Step 4,  Cost: 0.753
    Step 5,  Cost: 0.715
    Step 6,  Cost: 0.684
    Step 7,  Cost: 0.659
    Step 8,  Cost: 0.638
    Step 9,  Cost: 0.622
    Step 10,  Cost: 0.608
    Step 11,  Cost: 0.598
    Step 12,  Cost: 0.589
    Step 13,  Cost: 0.582
    Step 14,  Cost: 0.577
    Step 15,  Cost: 0.572
    Step 16,  Cost: 0.568
    Step 17,  Cost: 0.565
    Step 18,  Cost: 0.563
    Step 19,  Cost: 0.561
    Step 20,  Cost: 0.559
    Step 21,  Cost: 0.558
    Step 22,  Cost: 0.557
    Step 23,  Cost: 0.556
    Step 24,  Cost: 0.555
    Step 25,  Cost: 0.555
    Step 26,  Cost: 0.554
    Step 27,  Cost: 0.554
    Step 28,  Cost: 0.553
    Step 29,  Cost: 0.553
    Step 30,  Cost: 0.553
    Step 31,  Cost: 0.552
    Step 32,  Cost: 0.552
    Step 33,  Cost: 0.552
    Step 34,  Cost: 0.552
    Step 35,  Cost: 0.551
    Step 36,  Cost: 0.551
    Step 37,  Cost: 0.551
    Step 38,  Cost: 0.551
    Step 39,  Cost: 0.551
    Step 40,  Cost: 0.551
    Step 41,  Cost: 0.551
    Step 42,  Cost: 0.551
    Step 43,  Cost: 0.551
    Step 44,  Cost: 0.551
    Step 45,  Cost: 0.551
    Step 46,  Cost: 0.550
    Step 47,  Cost: 0.550
    Step 48,  Cost: 0.550
    Step 49,  Cost: 0.550
    Step 50,  Cost: 0.550
    Step 51,  Cost: 0.550
    Step 52,  Cost: 0.550
    Step 53,  Cost: 0.550
    Step 54,  Cost: 0.550
    Step 55,  Cost: 0.550
    Step 56,  Cost: 0.550
    Step 57,  Cost: 0.550
    Step 58,  Cost: 0.550
    Step 59,  Cost: 0.550
    Step 60,  Cost: 0.550
    Step 61,  Cost: 0.550
    Step 62,  Cost: 0.550
    Step 63,  Cost: 0.550
    Step 64,  Cost: 0.550
    Step 65,  Cost: 0.550
    Step 66,  Cost: 0.550
    Step 67,  Cost: 0.550
    Step 68,  Cost: 0.550
    Step 69,  Cost: 0.550
    Step 70,  Cost: 0.550
    Step 71,  Cost: 0.550
    Step 72,  Cost: 0.550
    Step 73,  Cost: 0.550
    Step 74,  Cost: 0.550
    Step 75,  Cost: 0.550
    Step 76,  Cost: 0.550
    Step 77,  Cost: 0.550
    Step 78,  Cost: 0.550
    Step 79,  Cost: 0.550
    Step 80,  Cost: 0.550
    Step 81,  Cost: 0.550
    Step 82,  Cost: 0.550
    Step 83,  Cost: 0.550
    Step 84,  Cost: 0.550
    Step 85,  Cost: 0.550
    Step 86,  Cost: 0.550
    Step 87,  Cost: 0.550
    Step 88,  Cost: 0.550
    Step 89,  Cost: 0.550
    Step 90,  Cost: 0.550
    Step 91,  Cost: 0.550
    Step 92,  Cost: 0.550
    Step 93,  Cost: 0.550
    Step 94,  Cost: 0.550
    Step 95,  Cost: 0.550
    Step 96,  Cost: 0.550
    Step 97,  Cost: 0.550
    Step 98,  Cost: 0.550
    Step 99,  Cost: 0.550
    Step 100,  Cost: 0.550
    예측값: [0 1 2 0 0 2]
    실제값: [0 1 2 0 0 2]
    정확도: 100.00
    

![image.png](attachment:image.png)

![image.png](attachment:image.png)


```python

```
