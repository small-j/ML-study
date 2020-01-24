{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Chapter 4.2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From c:\\users\\rlawl\\appdata\\local\\programs\\python\\python37\\lib\\site-packages\\tensorflow_core\\python\\compat\\v2_compat.py:65: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "non-resource variables are not supported in the long term\n"
     ]
    }
   ],
   "source": [
    "import tensorflow.compat.v1 as tf\n",
    "import numpy as np\n",
    "tf.disable_v2_behavior()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "#[털, 날개]\n",
    "x_data = np.array(\n",
    "[[0,0],[1,0],[1,1],[0,0],[0,0],[0,1]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "#기타 = [1,0,0]\n",
    "#포유류 = [0,1,0]\n",
    "#조류 = [0,0,1]\n",
    "#인덱스의 값만 1로 설정하고 나머지는 0으로 채우는것 => 원-핫-인코딩"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_data = np.array([\n",
    "    [1,0,0],  #기타\n",
    "    [0,1,0],  #포유류\n",
    "    [0,0,1],  #조류\n",
    "    [1,0,0],\n",
    "    [1,0,0],\n",
    "    [0,0,1]\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = tf.placeholder(tf.float32)\n",
    "Y = tf.placeholder(tf.float32)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "W = tf.Variable(tf.random_uniform([2,3], -1., 1.))\n",
    "b = tf.Variable(tf.zeros([3]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "L = tf.add(tf.matmul(X, W), b)\n",
    "L = tf.nn.relu(L)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = tf.nn.softmax(L)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 기본적인 경사하강법으로 최적화합니다.\n",
    "optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)\n",
    "train_op = optimizer.minimize(cost)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 텐서플로의 세션을 초기화합니다.\n",
    "init = tf.global_variables_initializer()\n",
    "sess = tf.Session()\n",
    "sess.run(init)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "10 0.96659666\n",
      "20 0.9632659\n",
      "30 0.95999926\n",
      "40 0.9568374\n",
      "50 0.95376843\n",
      "60 0.95068103\n",
      "70 0.947736\n",
      "80 0.94479394\n",
      "90 0.94199467\n",
      "100 0.93920517\n",
      "예측값: [0 2 2 0 0 2]\n",
      "실제값: [0 1 2 0 0 2]\n",
      "정확도: 83.33\n"
     ]
    }
   ],
   "source": [
    "# 앞서 구성한 특징과 레이블 데이터를 이용해 학습을 100번 진행합니다.\n",
    "for step in range(100):\n",
    "    sess.run(train_op, feed_dict ={X: x_data, Y: y_data})\n",
    "    \n",
    "    #학습 도중 10번에 한 번씩 손실값을 출력해봅니다.\n",
    "    if(step + 1) % 10 == 0:\n",
    "        print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))\n",
    "\n",
    "prediction = tf.argmax(model, axis=1)\n",
    "target = tf.argmax(Y, axis=1)\n",
    "print('예측값:', sess.run(prediction, feed_dict={X: x_data}))\n",
    "print('실제값:', sess.run(target, feed_dict={Y: y_data}))\n",
    "\n",
    "is_correct = tf.equal(prediction, target)\n",
    "accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))\n",
    "print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Chapter 4.3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 가중치\n",
    "W1 = tf.Variable(tf.random_uniform([2,10], -1., 1.))\n",
    "W2 = tf.Variable(tf.random_uniform([10,3], -1., 1.))\n",
    "\n",
    "# 편향\n",
    "b1 = tf.Variable(tf.zeros([10]))\n",
    "b2 = tf.Variable(tf.zeros([3]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "L1 = tf.add(tf.matmul(X, W1), b1)\n",
    "L1 = tf.nn.relu(L1)\n",
    "\n",
    "model = tf.add(tf.matmul(L1, W2), b2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From <ipython-input-15-dcafe696630e>:2: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.\n",
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
    "cost = tf.reduce_mean(\n",
    "tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))\n",
    "\n",
    "optimizer = tf.train.AdamOptimizer(learning_rate=0.01)\n",
    "train_op = optimizer.minimize(cost)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "10 0.7387759\n",
      "20 0.5472246\n",
      "30 0.3963021\n",
      "40 0.2675613\n",
      "50 0.16909295\n",
      "60 0.10465491\n",
      "70 0.06615349\n",
      "80 0.044335317\n",
      "90 0.03169268\n",
      "100 0.023941787\n",
      "예측값: [0 1 2 0 0 2]\n",
      "실제값: [0 1 2 0 0 2]\n",
      "정확도: 100.00\n"
     ]
    }
   ],
   "source": [
    "# 텐서플로의 세션을 초기화합니다.\n",
    "init = tf.global_variables_initializer()\n",
    "sess = tf.Session()\n",
    "sess.run(init)\n",
    "\n",
    "# 앞서 구성한 특징과 레이블 데이터를 이용해 학습을 100번 진행합니다.\n",
    "for step in range(100):\n",
    "    sess.run(train_op, feed_dict ={X: x_data, Y: y_data})\n",
    "    \n",
    "    #학습 도중 10번에 한 번씩 손실값을 출력해봅니다.\n",
    "    if(step + 1) % 10 == 0:\n",
    "        print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))\n",
    "\n",
    "prediction = tf.argmax(model, axis=1)\n",
    "target = tf.argmax(Y, axis=1)\n",
    "print('예측값:', sess.run(prediction, feed_dict={X: x_data}))\n",
    "print('실제값:', sess.run(target, feed_dict={Y: y_data}))\n",
    "\n",
    "is_correct = tf.equal(prediction, target)\n",
    "accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))\n",
    "print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))"
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
