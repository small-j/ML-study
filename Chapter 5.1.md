{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Chapter 5.1"
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
      "non-resource variables are not supported in the long term\n"
     ]
    }
   ],
   "source": [
    "import tensorflow.compat.v1 as tf\n",
    "import numpy as np\n",
    "tf.disable_v2_behavior()\n",
    "\n",
    "data = np.loadtxt('./data.csv', delimiter=',',\n",
    "                 unpack=True, dtype='float32')\n",
    "\n",
    "x_data = np.transpose(data[0:2])\n",
    "y_data = np.transpose(data[2:])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "### 신경망 모델 구성\n",
    "global_step = tf.Variable(0, trainable=False, name='global_step')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = tf.placeholder(tf.float32)\n",
    "Y = tf.placeholder(tf.float32)\n",
    "\n",
    "\n",
    "W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))\n",
    "L1 = tf.nn.relu(tf.matmul(X, W1))\n",
    "\n",
    "\n",
    "W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.))\n",
    "L2 = tf.nn.relu(tf.matmul(L1, W2))\n",
    "\n",
    "W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.))\n",
    "model = tf.matmul(L2, W3)\n",
    "\n",
    "cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))\n",
    "\n",
    "optimizer = tf.train.AdamOptimizer(learning_rate=0.01)\n",
    "train_op = optimizer.minimize(cost, global_step=global_step)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "### 신경망 모델 학습\n",
    "sess = tf.Session()\n",
    "saver = tf.train.Saver(tf.global_variables())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "ckpt = tf.train.get_checkpoint_state('./model')\n",
    "if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):\n",
    "    saver.restore(sess, ckpt.model_checkpoint_path)\n",
    "else:\n",
    "    sess.run(tf.global_variables_initializer())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Step 101,  Cost: 0.550\n",
      "Step 102,  Cost: 0.550\n"
     ]
    }
   ],
   "source": [
    "for step in range(2):\n",
    "    sess.run(train_op, feed_dict={X: x_data, Y: y_data})\n",
    "\n",
    "    print('Step %d, '% sess.run(global_step),\n",
    "     'Cost: %.3f' % sess.run(cost, feed_dict={X: x_data, Y: y_data}))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'./model/dnn.ckpt-100'"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "saver.save(sess, './model/dnn.ckpt', global_step=global_step)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "예측값: [0 1 2 0 0 2]\n",
      "실제값: [0 1 2 0 0 2]\n",
      "정확도: 100.00\n"
     ]
    }
   ],
   "source": [
    "###결과확인\n",
    "prediction = tf.argmax(model, 1)\n",
    "target = tf.argmax(Y, 1)\n",
    "print('예측값:', sess.run(prediction, feed_dict={X: x_data}))\n",
    "print('실제값:', sess.run(target, feed_dict={Y: y_data}))\n",
    "\n",
    "is_correct = tf.equal(prediction, target)\n",
    "accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))\n",
    "print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))"
   ]
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