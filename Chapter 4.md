## Chapter 4.2


```python
import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()
```

    WARNING:tensorflow:From c:\users\rlawl\appdata\local\programs\python\python37\lib\site-packages\tensorflow_core\python\compat\v2_compat.py:65: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
    Instructions for updating:
    non-resource variables are not supported in the long term
    


```python
#[털, 날개]
x_data = np.array(
[[0,0],[1,0],[1,1],[0,0],[0,0],[0,1]])
```


```python
#기타 = [1,0,0]
#포유류 = [0,1,0]
#조류 = [0,0,1]
#인덱스의 값만 1로 설정하고 나머지는 0으로 채우는것 => 원-핫-인코딩
```


```python
y_data = np.array([
    [1,0,0],  #기타
    [0,1,0],  #포유류
    [0,0,1],  #조류
    [1,0,0],
    [1,0,0],
    [0,0,1]
])
```


```python
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
```


```python
W = tf.Variable(tf.random_uniform([2,3], -1., 1.))
b = tf.Variable(tf.zeros([3]))
```


```python
L = tf.add(tf.matmul(X, W), b)
L = tf.nn.relu(L)
```


```python
model = tf.nn.softmax(L)
```


```python
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))
```


```python
# 기본적인 경사하강법으로 최적화합니다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)
```


```python
# 텐서플로의 세션을 초기화합니다.
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
```


```python
# 앞서 구성한 특징과 레이블 데이터를 이용해 학습을 100번 진행합니다.
for step in range(100):
    sess.run(train_op, feed_dict ={X: x_data, Y: y_data})
    
    #학습 도중 10번에 한 번씩 손실값을 출력해봅니다.
    if(step + 1) % 10 == 0:
        print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y, axis=1)
print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
print('실제값:', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
```

    10 0.96659666
    20 0.9632659
    30 0.95999926
    40 0.9568374
    50 0.95376843
    60 0.95068103
    70 0.947736
    80 0.94479394
    90 0.94199467
    100 0.93920517
    예측값: [0 2 2 0 0 2]
    실제값: [0 1 2 0 0 2]
    정확도: 83.33
    

## Chapter 4.3


```python
# 가중치
W1 = tf.Variable(tf.random_uniform([2,10], -1., 1.))
W2 = tf.Variable(tf.random_uniform([10,3], -1., 1.))

# 편향
b1 = tf.Variable(tf.zeros([10]))
b2 = tf.Variable(tf.zeros([3]))
```


```python
L1 = tf.add(tf.matmul(X, W1), b1)
L1 = tf.nn.relu(L1)

model = tf.add(tf.matmul(L1, W2), b2)
```


```python
cost = tf.reduce_mean(
tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)
```

    WARNING:tensorflow:From <ipython-input-15-dcafe696630e>:2: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    
    Future major versions of TensorFlow will allow gradients to flow
    into the labels input on backprop by default.
    
    See `tf.nn.softmax_cross_entropy_with_logits_v2`.
    
    


```python
# 텐서플로의 세션을 초기화합니다.
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 앞서 구성한 특징과 레이블 데이터를 이용해 학습을 100번 진행합니다.
for step in range(100):
    sess.run(train_op, feed_dict ={X: x_data, Y: y_data})
    
    #학습 도중 10번에 한 번씩 손실값을 출력해봅니다.
    if(step + 1) % 10 == 0:
        print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y, axis=1)
print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
print('실제값:', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
```

    10 0.7387759
    20 0.5472246
    30 0.3963021
    40 0.2675613
    50 0.16909295
    60 0.10465491
    70 0.06615349
    80 0.044335317
    90 0.03169268
    100 0.023941787
    예측값: [0 1 2 0 0 2]
    실제값: [0 1 2 0 0 2]
    정확도: 100.00
    


```python

```
