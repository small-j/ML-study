## Chapter 6.1


```python
import tensorflow.compat.v1 as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.disable_v2_behavior()
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)  #mnist 데이터를 내려받고 원-핫 인코딩 방식으로 읽어드림
```

    WARNING:tensorflow:From c:\users\rlawl\appdata\local\programs\python\python37\lib\site-packages\tensorflow_core\python\compat\v2_compat.py:65: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
    Instructions for updating:
    non-resource variables are not supported in the long term
    WARNING:tensorflow:From <ipython-input-1-bc7ec113bd3e>:4: read_data_sets (from tensorflow.examples.tutorials.mnist.input_data) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use alternatives such as: tensorflow_datasets.load('mnist')
    WARNING:tensorflow:From c:\users\rlawl\appdata\local\programs\python\python37\lib\site-packages\tensorflow_core\examples\tutorials\mnist\input_data.py:297: _maybe_download (from tensorflow.examples.tutorials.mnist.input_data) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please write your own downloading logic.
    WARNING:tensorflow:From c:\users\rlawl\appdata\local\programs\python\python37\lib\site-packages\tensorflow_core\examples\tutorials\mnist\input_data.py:299: _extract_images (from tensorflow.examples.tutorials.mnist.input_data) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use tf.data to implement this functionality.
    Extracting ./mnist/data/train-images-idx3-ubyte.gz
    WARNING:tensorflow:From c:\users\rlawl\appdata\local\programs\python\python37\lib\site-packages\tensorflow_core\examples\tutorials\mnist\input_data.py:304: _extract_labels (from tensorflow.examples.tutorials.mnist.input_data) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use tf.data to implement this functionality.
    Extracting ./mnist/data/train-labels-idx1-ubyte.gz
    WARNING:tensorflow:From c:\users\rlawl\appdata\local\programs\python\python37\lib\site-packages\tensorflow_core\examples\tutorials\mnist\input_data.py:112: _dense_to_one_hot (from tensorflow.examples.tutorials.mnist.input_data) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use tf.one_hot on tensors.
    Extracting ./mnist/data/t10k-images-idx3-ubyte.gz
    Extracting ./mnist/data/t10k-labels-idx1-ubyte.gz
    WARNING:tensorflow:From c:\users\rlawl\appdata\local\programs\python\python37\lib\site-packages\tensorflow_core\examples\tutorials\mnist\input_data.py:328: _DataSet.__init__ (from tensorflow.examples.tutorials.mnist.input_data) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use alternatives such as official/mnist/_DataSet.py from tensorflow/models.
    


```python
X = tf.placeholder(tf.float32, [None, 784])   #28x28 픽셀의 손글씨 이미지는 784개의 특징으로 이루어져 있음
Y = tf.placeholder(tf.float32, [None, 10])    #레이블은 0~9까지 
#첫번째 차원이 None인 이유는 한번에 학습시킬 mnist 이미지의 개수를 지정하는 값(텐서플로가 알아서 계산)
```


```python
W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))

W2 = tf.Variable(tf.random_normal([256,256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L2, W3)
```


```python
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
```

    WARNING:tensorflow:From <ipython-input-4-b021224dac13>:1: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    
    Future major versions of TensorFlow will allow gradients to flow
    into the labels input on backprop by default.
    
    See `tf.nn.softmax_cross_entropy_with_logits_v2`.
    
    


```python
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
```


```python
batch_size = 100     # MNIST가 데이터가 매우 크므로 학습에 미니배치를 사용할 것이다. 미니배치 크기 : 100
total_batch = int(mnist.train.num_examples / batch_size)   #학습 데이터의 총 개수를 배치크기로 나누어 미니배치가 몇개인지
```


```python
for epoch in range(15):
    total_cost = 0
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)   #알아서 값 배치(배치크기만큼) 입력값은 xs, 출력값(레이블데이터) ys
        
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})  #손실값을 저장, feed_dict 매개변수에 
        total_cost += cost_val
        
    print('Epoch:', '%04d' % (epoch + 1), 
         'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))    # 한 세대의 학습이 끝나면 학습한 세대의 평균 손실값 출력

print('최적화 완료!')
```

    Epoch: 0001 Avg. cost = 0.006
    Epoch: 0002 Avg. cost = 0.001
    Epoch: 0003 Avg. cost = 0.005
    Epoch: 0004 Avg. cost = 0.008
    Epoch: 0005 Avg. cost = 0.004
    Epoch: 0006 Avg. cost = 0.001
    Epoch: 0007 Avg. cost = 0.001
    Epoch: 0008 Avg. cost = 0.009
    Epoch: 0009 Avg. cost = 0.007
    Epoch: 0010 Avg. cost = 0.003
    Epoch: 0011 Avg. cost = 0.004
    Epoch: 0012 Avg. cost = 0.001
    Epoch: 0013 Avg. cost = 0.001
    Epoch: 0014 Avg. cost = 0.003
    Epoch: 0015 Avg. cost = 0.008
    최적화 완료!
    


```python
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))  #예측한 결과 : model 과 실제 레이블 : Y 비교 -> 학습이 잘 되었나 확인 
#  인덱스에 들어가는 값은 해당 숫자가 얼마나 해당 인덱스와 관련이 높은가를 나타낸다. 값이 가장 큰 인덱스가 가장 근접한 예측 결과
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))  #tf.cast 로 is_correct를 0과 1로 변환 , reduce_mean으로 평균계산

print('정확도:', sess.run(accuracy,
                      feed_dict={X: mnist.test.images, Y: mnist.test.labels})) 
#테스트 데이터를 다루는 객체인 mnist.test 를 이용해 이미지와 레이블 데이터를 넣어 accuracy 계산 
```

    정확도: 0.9794
    

## Chapter 6.2
###  드롭아웃

#### 과적합 : 학습한 결과가 학습 데이터에 너무 맞춰져있어 다른 데이터에 잘 맞지 않은 상황 이 문제를 해결한 것이 드롭아웃 

#### 학습시 전체 신경망중 일부만 사용(일부 뉴런을 제거하여 일부 특징이 특정 뉴런에 고정되는 것을 막음, 신경망이 학습되기까지 더 오래걸림)


```python
X = tf.placeholder(tf.float32, [None, 784])   #28x28 픽셀의 손글씨 이미지는 784개의 특징으로 이루어져 있음
Y = tf.placeholder(tf.float32, [None, 10])    #레이블은 0~9까지 
#첫번째 차원이 None인 이유는 한번에 학습시킬 mnist 이미지의 개수를 지정하는 값(텐서플로가 알아서 계산)
```


```python
keep_prob = tf.placeholder(tf.float32)  #학습시에는 0.8, 예측 시에는 1을 넣어 신경망 전체를 사용하도록 만듬

W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))
L1 = tf.nn.dropout(L1, keep_prob)
#L1 = tf.nn.dropout(L1, 0.8)   #사용할 뉴런의 비율 (80%만 사용하겠다는 것)

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))
L2 = tf.nn.dropout(L2, keep_prob)
#L2 = tf.nn.dropout(L2, 0.8)

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L2, W3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
```


```python
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100     # MNIST가 데이터가 매우 크므로 학습에 미니배치를 사용할 것이다. 미니배치 크기 : 100
total_batch = int(mnist.train.num_examples / batch_size)   #학습 데이터의 총 개수를 배치크기로 나누어 미니배치가 몇개인지

for epoch in range(15):
    total_cost = 0
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)   #알아서 값 배치(배치크기만큼) 입력값은 xs, 출력값(레이블데이터) ys
        
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.8})  #keep_prob을 0.8로
        total_cost += cost_val
        
    print('Epoch:', '%04d' % (epoch + 1), 
         'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))    # 한 세대의 학습이 끝나면 학습한 세대의 평균 손실값 출력

print('최적화 완료!')
```

    Epoch: 0001 Avg. cost = 0.422
    Epoch: 0002 Avg. cost = 0.160
    Epoch: 0003 Avg. cost = 0.113
    Epoch: 0004 Avg. cost = 0.087
    Epoch: 0005 Avg. cost = 0.072
    Epoch: 0006 Avg. cost = 0.062
    Epoch: 0007 Avg. cost = 0.052
    Epoch: 0008 Avg. cost = 0.048
    Epoch: 0009 Avg. cost = 0.041
    Epoch: 0010 Avg. cost = 0.038
    Epoch: 0011 Avg. cost = 0.035
    Epoch: 0012 Avg. cost = 0.031
    Epoch: 0013 Avg. cost = 0.027
    Epoch: 0014 Avg. cost = 0.027
    Epoch: 0015 Avg. cost = 0.025
    최적화 완료!
    


```python
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))  #예측한 결과 : model 과 실제 레이블 : Y 비교 -> 학습이 잘 되었나 확인 
#  인덱스에 들어가는 값은 해당 숫자가 얼마나 해당 인덱스와 관련이 높은가를 나타낸다. 값이 가장 큰 인덱스가 가장 근접한 예측 결과
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))  #tf.cast 로 is_correct를 0과 1로 변환 , reduce_mean으로 평균계산

print('정확도:', sess.run(accuracy,
                      feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))  #keep_prob을 1로
```

    정확도: 0.9824
    

## Chapter 6.3


```python
import matplotlib.pyplot as plt
import numpy as np

X = tf.placeholder(tf.float32, [None, 784])   
Y = tf.placeholder(tf.float32, [None, 10])    

keep_prob = tf.placeholder(tf.float32)  #학습시에는 0.8, 예측 시에는 1을 넣어 신경망 전체를 사용하도록 만듬

W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))
L1 = tf.nn.dropout(L1, keep_prob)
#L1 = tf.nn.dropout(L1, 0.8)   #사용할 뉴런의 비율 (80%만 사용하겠다는 것)

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))
L2 = tf.nn.dropout(L2, keep_prob)
#L2 = tf.nn.dropout(L2, 0.8)

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L2, W3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100     # MNIST가 데이터가 매우 크므로 학습에 미니배치를 사용할 것이다. 미니배치 크기 : 100
total_batch = int(mnist.train.num_examples / batch_size)   #학습 데이터의 총 개수를 배치크기로 나누어 미니배치가 몇개인지

for epoch in range(15):
    total_cost = 0
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)   #알아서 값 배치(배치크기만큼) 입력값은 xs, 출력값(레이블데이터) ys
        
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.8})  #keep_prob을 0.8로
        total_cost += cost_val
        
    print('Epoch:', '%04d' % (epoch + 1), 
         'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))    # 한 세대의 학습이 끝나면 학습한 세대의 평균 손실값 출력

print('최적화 완료!')

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))  #예측한 결과 : model 과 실제 레이블 : Y 비교 -> 학습이 잘 되었나 확인 
#  인덱스에 들어가는 값은 해당 숫자가 얼마나 해당 인덱스와 관련이 높은가를 나타낸다. 값이 가장 큰 인덱스가 가장 근접한 예측 결과
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))  #tf.cast 로 is_correct를 0과 1로 변환 , reduce_mean으로 평균계산

print('정확도:', sess.run(accuracy,
                      feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))  #keep_prob을 1로
```

    Epoch: 0001 Avg. cost = 0.443
    Epoch: 0002 Avg. cost = 0.171
    Epoch: 0003 Avg. cost = 0.120
    Epoch: 0004 Avg. cost = 0.091
    Epoch: 0005 Avg. cost = 0.077
    Epoch: 0006 Avg. cost = 0.063
    Epoch: 0007 Avg. cost = 0.054
    Epoch: 0008 Avg. cost = 0.048
    Epoch: 0009 Avg. cost = 0.042
    Epoch: 0010 Avg. cost = 0.039
    Epoch: 0011 Avg. cost = 0.035
    Epoch: 0012 Avg. cost = 0.032
    Epoch: 0013 Avg. cost = 0.031
    Epoch: 0014 Avg. cost = 0.028
    Epoch: 0015 Avg. cost = 0.027
    최적화 완료!
    정확도: 0.9822
    


```python
# matplotlib 결과확인
labels = sess.run(model, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob:1})
```


```python
fig = plt.figure()   #손글씨 출력할 그래프 준비
for i in range(10):
    subplot = fig.add_subplot(2, 5, i + 1)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title('%d' % np.argmax(labels[i]))
    subplot.imshow(mnist.test.images[i].reshape((28,28)),
                  cmap=plt.cm.gray_r)
    
plt.show()
```


![png](output_19_0.png)


### 예측한 결과가 일치하는 것을 볼 수 있다.


```python

```
