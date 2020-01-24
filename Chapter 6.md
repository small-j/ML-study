{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Chapter 6.1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From c:\\users\\rlawl\\appdata\\local\\programs\\python\\python37\\lib\\site-packages\\tensorflow_core\\python\\compat\\v2_compat.py:65: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "non-resource variables are not supported in the long term\n",
      "WARNING:tensorflow:From <ipython-input-1-bc7ec113bd3e>:4: read_data_sets (from tensorflow.examples.tutorials.mnist.input_data) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use alternatives such as: tensorflow_datasets.load('mnist')\n",
      "WARNING:tensorflow:From c:\\users\\rlawl\\appdata\\local\\programs\\python\\python37\\lib\\site-packages\\tensorflow_core\\examples\\tutorials\\mnist\\input_data.py:297: _maybe_download (from tensorflow.examples.tutorials.mnist.input_data) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please write your own downloading logic.\n",
      "WARNING:tensorflow:From c:\\users\\rlawl\\appdata\\local\\programs\\python\\python37\\lib\\site-packages\\tensorflow_core\\examples\\tutorials\\mnist\\input_data.py:299: _extract_images (from tensorflow.examples.tutorials.mnist.input_data) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use tf.data to implement this functionality.\n",
      "Extracting ./mnist/data/train-images-idx3-ubyte.gz\n",
      "WARNING:tensorflow:From c:\\users\\rlawl\\appdata\\local\\programs\\python\\python37\\lib\\site-packages\\tensorflow_core\\examples\\tutorials\\mnist\\input_data.py:304: _extract_labels (from tensorflow.examples.tutorials.mnist.input_data) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use tf.data to implement this functionality.\n",
      "Extracting ./mnist/data/train-labels-idx1-ubyte.gz\n",
      "WARNING:tensorflow:From c:\\users\\rlawl\\appdata\\local\\programs\\python\\python37\\lib\\site-packages\\tensorflow_core\\examples\\tutorials\\mnist\\input_data.py:112: _dense_to_one_hot (from tensorflow.examples.tutorials.mnist.input_data) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use tf.one_hot on tensors.\n",
      "Extracting ./mnist/data/t10k-images-idx3-ubyte.gz\n",
      "Extracting ./mnist/data/t10k-labels-idx1-ubyte.gz\n",
      "WARNING:tensorflow:From c:\\users\\rlawl\\appdata\\local\\programs\\python\\python37\\lib\\site-packages\\tensorflow_core\\examples\\tutorials\\mnist\\input_data.py:328: _DataSet.__init__ (from tensorflow.examples.tutorials.mnist.input_data) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use alternatives such as official/mnist/_DataSet.py from tensorflow/models.\n"
     ]
    }
   ],
   "source": [
    "import tensorflow.compat.v1 as tf\n",
    "from tensorflow.examples.tutorials.mnist import input_data\n",
    "tf.disable_v2_behavior()\n",
    "mnist = input_data.read_data_sets(\"./mnist/data/\", one_hot=True)  #mnist 데이터를 내려받고 원-핫 인코딩 방식으로 읽어드림"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = tf.placeholder(tf.float32, [None, 784])   #28x28 픽셀의 손글씨 이미지는 784개의 특징으로 이루어져 있음\n",
    "Y = tf.placeholder(tf.float32, [None, 10])    #레이블은 0~9까지 \n",
    "#첫번째 차원이 None인 이유는 한번에 학습시킬 mnist 이미지의 개수를 지정하는 값(텐서플로가 알아서 계산)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))\n",
    "L1 = tf.nn.relu(tf.matmul(X, W1))\n",
    "\n",
    "W2 = tf.Variable(tf.random_normal([256,256], stddev=0.01))\n",
    "L2 = tf.nn.relu(tf.matmul(L1, W2))\n",
    "\n",
    "W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))\n",
    "model = tf.matmul(L2, W3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From <ipython-input-4-b021224dac13>:1: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "\n",
      "Future major versions of TensorFlow will allow gradients to flow\n",
      "into the labels input on backprop by default.\n",
      "\n",
      "See `tf.nn.softmax_cross_entropy_with_logits_v2`.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))\n",
    "optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "init = tf.global_variables_initializer()\n",
    "sess = tf.Session()\n",
    "sess.run(init)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "batch_size = 100     # MNIST가 데이터가 매우 크므로 학습에 미니배치를 사용할 것이다. 미니배치 크기 : 100\n",
    "total_batch = int(mnist.train.num_examples / batch_size)   #학습 데이터의 총 개수를 배치크기로 나누어 미니배치가 몇개인지"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0001 Avg. cost = 0.006\n",
      "Epoch: 0002 Avg. cost = 0.001\n",
      "Epoch: 0003 Avg. cost = 0.005\n",
      "Epoch: 0004 Avg. cost = 0.008\n",
      "Epoch: 0005 Avg. cost = 0.004\n",
      "Epoch: 0006 Avg. cost = 0.001\n",
      "Epoch: 0007 Avg. cost = 0.001\n",
      "Epoch: 0008 Avg. cost = 0.009\n",
      "Epoch: 0009 Avg. cost = 0.007\n",
      "Epoch: 0010 Avg. cost = 0.003\n",
      "Epoch: 0011 Avg. cost = 0.004\n",
      "Epoch: 0012 Avg. cost = 0.001\n",
      "Epoch: 0013 Avg. cost = 0.001\n",
      "Epoch: 0014 Avg. cost = 0.003\n",
      "Epoch: 0015 Avg. cost = 0.008\n",
      "최적화 완료!\n"
     ]
    }
   ],
   "source": [
    "for epoch in range(15):\n",
    "    total_cost = 0\n",
    "    for i in range(total_batch):\n",
    "        batch_xs, batch_ys = mnist.train.next_batch(batch_size)   #알아서 값 배치(배치크기만큼) 입력값은 xs, 출력값(레이블데이터) ys\n",
    "        \n",
    "        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})  #손실값을 저장, feed_dict 매개변수에 \n",
    "        total_cost += cost_val\n",
    "        \n",
    "    print('Epoch:', '%04d' % (epoch + 1), \n",
    "         'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))    # 한 세대의 학습이 끝나면 학습한 세대의 평균 손실값 출력\n",
    "\n",
    "print('최적화 완료!')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "정확도: 0.9794\n"
     ]
    }
   ],
   "source": [
    "is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))  #예측한 결과 : model 과 실제 레이블 : Y 비교 -> 학습이 잘 되었나 확인 \n",
    "#  인덱스에 들어가는 값은 해당 숫자가 얼마나 해당 인덱스와 관련이 높은가를 나타낸다. 값이 가장 큰 인덱스가 가장 근접한 예측 결과\n",
    "accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))  #tf.cast 로 is_correct를 0과 1로 변환 , reduce_mean으로 평균계산\n",
    "\n",
    "print('정확도:', sess.run(accuracy,\n",
    "                      feed_dict={X: mnist.test.images, Y: mnist.test.labels})) \n",
    "#테스트 데이터를 다루는 객체인 mnist.test 를 이용해 이미지와 레이블 데이터를 넣어 accuracy 계산 "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Chapter 6.2\n",
    "###  드롭아웃"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 과적합 : 학습한 결과가 학습 데이터에 너무 맞춰져있어 다른 데이터에 잘 맞지 않은 상황 이 문제를 해결한 것이 드롭아웃 "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 학습시 전체 신경망중 일부만 사용(일부 뉴런을 제거하여 일부 특징이 특정 뉴런에 고정되는 것을 막음, 신경망이 학습되기까지 더 오래걸림)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = tf.placeholder(tf.float32, [None, 784])   #28x28 픽셀의 손글씨 이미지는 784개의 특징으로 이루어져 있음\n",
    "Y = tf.placeholder(tf.float32, [None, 10])    #레이블은 0~9까지 \n",
    "#첫번째 차원이 None인 이유는 한번에 학습시킬 mnist 이미지의 개수를 지정하는 값(텐서플로가 알아서 계산)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "keep_prob = tf.placeholder(tf.float32)  #학습시에는 0.8, 예측 시에는 1을 넣어 신경망 전체를 사용하도록 만듬\n",
    "\n",
    "W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))\n",
    "L1 = tf.nn.relu(tf.matmul(X, W1))\n",
    "L1 = tf.nn.dropout(L1, keep_prob)\n",
    "#L1 = tf.nn.dropout(L1, 0.8)   #사용할 뉴런의 비율 (80%만 사용하겠다는 것)\n",
    "\n",
    "W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))\n",
    "L2 = tf.nn.relu(tf.matmul(L1, W2))\n",
    "L2 = tf.nn.dropout(L2, keep_prob)\n",
    "#L2 = tf.nn.dropout(L2, 0.8)\n",
    "\n",
    "W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))\n",
    "model = tf.matmul(L2, W3)\n",
    "\n",
    "cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))\n",
    "optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0001 Avg. cost = 0.422\n",
      "Epoch: 0002 Avg. cost = 0.160\n",
      "Epoch: 0003 Avg. cost = 0.113\n",
      "Epoch: 0004 Avg. cost = 0.087\n",
      "Epoch: 0005 Avg. cost = 0.072\n",
      "Epoch: 0006 Avg. cost = 0.062\n",
      "Epoch: 0007 Avg. cost = 0.052\n",
      "Epoch: 0008 Avg. cost = 0.048\n",
      "Epoch: 0009 Avg. cost = 0.041\n",
      "Epoch: 0010 Avg. cost = 0.038\n",
      "Epoch: 0011 Avg. cost = 0.035\n",
      "Epoch: 0012 Avg. cost = 0.031\n",
      "Epoch: 0013 Avg. cost = 0.027\n",
      "Epoch: 0014 Avg. cost = 0.027\n",
      "Epoch: 0015 Avg. cost = 0.025\n",
      "최적화 완료!\n"
     ]
    }
   ],
   "source": [
    "init = tf.global_variables_initializer()\n",
    "sess = tf.Session()\n",
    "sess.run(init)\n",
    "\n",
    "batch_size = 100     # MNIST가 데이터가 매우 크므로 학습에 미니배치를 사용할 것이다. 미니배치 크기 : 100\n",
    "total_batch = int(mnist.train.num_examples / batch_size)   #학습 데이터의 총 개수를 배치크기로 나누어 미니배치가 몇개인지\n",
    "\n",
    "for epoch in range(15):\n",
    "    total_cost = 0\n",
    "    for i in range(total_batch):\n",
    "        batch_xs, batch_ys = mnist.train.next_batch(batch_size)   #알아서 값 배치(배치크기만큼) 입력값은 xs, 출력값(레이블데이터) ys\n",
    "        \n",
    "        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.8})  #keep_prob을 0.8로\n",
    "        total_cost += cost_val\n",
    "        \n",
    "    print('Epoch:', '%04d' % (epoch + 1), \n",
    "         'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))    # 한 세대의 학습이 끝나면 학습한 세대의 평균 손실값 출력\n",
    "\n",
    "print('최적화 완료!')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "정확도: 0.9824\n"
     ]
    }
   ],
   "source": [
    "is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))  #예측한 결과 : model 과 실제 레이블 : Y 비교 -> 학습이 잘 되었나 확인 \n",
    "#  인덱스에 들어가는 값은 해당 숫자가 얼마나 해당 인덱스와 관련이 높은가를 나타낸다. 값이 가장 큰 인덱스가 가장 근접한 예측 결과\n",
    "accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))  #tf.cast 로 is_correct를 0과 1로 변환 , reduce_mean으로 평균계산\n",
    "\n",
    "print('정확도:', sess.run(accuracy,\n",
    "                      feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))  #keep_prob을 1로"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Chapter 6.3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0001 Avg. cost = 0.443\n",
      "Epoch: 0002 Avg. cost = 0.171\n",
      "Epoch: 0003 Avg. cost = 0.120\n",
      "Epoch: 0004 Avg. cost = 0.091\n",
      "Epoch: 0005 Avg. cost = 0.077\n",
      "Epoch: 0006 Avg. cost = 0.063\n",
      "Epoch: 0007 Avg. cost = 0.054\n",
      "Epoch: 0008 Avg. cost = 0.048\n",
      "Epoch: 0009 Avg. cost = 0.042\n",
      "Epoch: 0010 Avg. cost = 0.039\n",
      "Epoch: 0011 Avg. cost = 0.035\n",
      "Epoch: 0012 Avg. cost = 0.032\n",
      "Epoch: 0013 Avg. cost = 0.031\n",
      "Epoch: 0014 Avg. cost = 0.028\n",
      "Epoch: 0015 Avg. cost = 0.027\n",
      "최적화 완료!\n",
      "정확도: 0.9822\n"
     ]
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "\n",
    "X = tf.placeholder(tf.float32, [None, 784])   \n",
    "Y = tf.placeholder(tf.float32, [None, 10])    \n",
    "\n",
    "keep_prob = tf.placeholder(tf.float32)  #학습시에는 0.8, 예측 시에는 1을 넣어 신경망 전체를 사용하도록 만듬\n",
    "\n",
    "W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))\n",
    "L1 = tf.nn.relu(tf.matmul(X, W1))\n",
    "L1 = tf.nn.dropout(L1, keep_prob)\n",
    "#L1 = tf.nn.dropout(L1, 0.8)   #사용할 뉴런의 비율 (80%만 사용하겠다는 것)\n",
    "\n",
    "W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))\n",
    "L2 = tf.nn.relu(tf.matmul(L1, W2))\n",
    "L2 = tf.nn.dropout(L2, keep_prob)\n",
    "#L2 = tf.nn.dropout(L2, 0.8)\n",
    "\n",
    "W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))\n",
    "model = tf.matmul(L2, W3)\n",
    "\n",
    "cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))\n",
    "optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)\n",
    "\n",
    "\n",
    "init = tf.global_variables_initializer()\n",
    "sess = tf.Session()\n",
    "sess.run(init)\n",
    "\n",
    "batch_size = 100     # MNIST가 데이터가 매우 크므로 학습에 미니배치를 사용할 것이다. 미니배치 크기 : 100\n",
    "total_batch = int(mnist.train.num_examples / batch_size)   #학습 데이터의 총 개수를 배치크기로 나누어 미니배치가 몇개인지\n",
    "\n",
    "for epoch in range(15):\n",
    "    total_cost = 0\n",
    "    for i in range(total_batch):\n",
    "        batch_xs, batch_ys = mnist.train.next_batch(batch_size)   #알아서 값 배치(배치크기만큼) 입력값은 xs, 출력값(레이블데이터) ys\n",
    "        \n",
    "        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.8})  #keep_prob을 0.8로\n",
    "        total_cost += cost_val\n",
    "        \n",
    "    print('Epoch:', '%04d' % (epoch + 1), \n",
    "         'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))    # 한 세대의 학습이 끝나면 학습한 세대의 평균 손실값 출력\n",
    "\n",
    "print('최적화 완료!')\n",
    "\n",
    "is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))  #예측한 결과 : model 과 실제 레이블 : Y 비교 -> 학습이 잘 되었나 확인 \n",
    "#  인덱스에 들어가는 값은 해당 숫자가 얼마나 해당 인덱스와 관련이 높은가를 나타낸다. 값이 가장 큰 인덱스가 가장 근접한 예측 결과\n",
    "accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))  #tf.cast 로 is_correct를 0과 1로 변환 , reduce_mean으로 평균계산\n",
    "\n",
    "print('정확도:', sess.run(accuracy,\n",
    "                      feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))  #keep_prob을 1로"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "# matplotlib 결과확인\n",
    "labels = sess.run(model, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob:1})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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Y5JbORwXvX9yq+pc0RSEiEgkGoBFRCLRACwiEokGYBGRSDQAi4hEUu4tHkXK0k8//QTA9OnTc369VatWlZ8PGJDaA2C77VK792y5ZWofy7Zt2xYyxLKgDFhEJJKSy4Aff/xxAA444AAABg0aBEDPnj0BaNSoUZzA8mDWrFkAHHHEEQDstttuAJx22mlAqmayPubOnVv5+X//+18A9tlnHwCaNGlSr9eWZBs/fjwQ3j/PP/88AJ9//nnO52+1VdjJfurUqQAsXpzZOve3337Lc5TlRxmwiEgkJZMB//DDD0DIdM3ZZ58NwMknnwzA6quvXtzA8sDm47bddlsgZKobbLABkL/Md6edwtZxc+bMAahchdS6det6fY9i+d///gfAJZdcAsCHH34IwIQJEwBl8l9++SUAgwcPBmDo0KEALFy4EICabsDw6afaQKMYlAGLiERSMhmwzVl+++23GY8fffTRADRt2rToMdWHZaAQ5nwty//73/8OhPnt+rrmmmsAmDJlSuVjlhmVSuY7cuRIAHr3Tm0NmH333jLj3/3ud8UNLGG++Sa1y/rAgQPr9O+33nprIFQ8NCRffPEFEN57Y8eOBcJ8+CqrpPLRM844Awj3YAr5HlEGLCISiQZgEZFIEj8FYaUvdhmd7fjjjwfAuRq3IE2Et956q/JzuwQyffr0ycv3+OCDDwDo378/AAcffHDl14488si8fI9Cs0vqXr16AeHyMfvnbTdjb7vtNgDWW2+9YoVYVPb/tymGjh07AqGccNVVVwVg7bXXBqBZs2YAzJ8/H4C9994bCFMMu+66KwA77rgjEG5ir7nmmgX8XxTH+++/D4QbkmPGjAFg9uzZK/13r776KhBu6FpJnh1rgH/9619AON51pQxYRCSSxGfA7733HpCZMQI0bpwKfd999y16TPVhiy0eeeSRFb42fPhwANZfv8abLORkme9ee+2V8fghhxxS+flaa61Vr+9RLJa92w3KqowaNQqAp556Cgg36ywzrm+mEtOCBQsqP7ef6bvvvgvAuHHjMp7boUMHAN5++20glDDaTcs//OEPQLjh1JDYWGEZ7+jRo4HMBUgQjkGnTp2AcIxuuukmAHbeeWcAXnvtNSCce08++WTla9gybLthV1cN76cgIlIiEp8B27xNtuzsrlRccMEFQCirgrBA4vDDD8/L93jppZcAmDlz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\n",
      "text/plain": [
       "<Figure size 432x288 with 10 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure()   #손글씨 출력할 그래프 준비\n",
    "for i in range(10):\n",
    "    subplot = fig.add_subplot(2, 5, i + 1)\n",
    "    subplot.set_xticks([])\n",
    "    subplot.set_yticks([])\n",
    "    subplot.set_title('%d' % np.argmax(labels[i]))\n",
    "    subplot.imshow(mnist.test.images[i].reshape((28,28)),\n",
    "                  cmap=plt.cm.gray_r)\n",
    "    \n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 예측한 결과가 일치하는 것을 볼 수 있다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
