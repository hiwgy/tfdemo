#! /Users/wangguangyuan/venv/bin/python3

import gym
import numpy as np
import tensorflow as tf

env = gym.make('CartPole-v0')
env.reset()
random_episodes = 0
reward_sum = 0
cnt = 0
while random_episodes < 10:
    env.render()
    observation, reward, done, _ = env.step(np.random.randint(0, 2))
    reward_sum += reward
    cnt += 1

    if done:
        random_episodes += 1
        print("Reward for this episode was: %d, action cnt was:%d" % (reward_sum, cnt))
        reward_sum = 0
        cnt = 0
        env.reset()

H = 50
batch_size = 25
learning_rate = 1e-1
D = 4
gamma = 0.99

observations = tf.placeholder(tf.float32, [None, D], name="input_x")
W1 = tf.get_variable("W1", shape=[D, H],
        initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations, W1))
W2 = tf.get_variable("W2", shape=[H, 1],
        initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1, W2)
probability = tf.nn.sigmoid(score)

adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
W1Grad = tf.placeholder(tf.float32, name="batch_grad1")
W2Grad = tf.placeholder(tf.float32, name="batch_grad2")
batchGrad = [W1Grad, W2Grad]
# fix a bug of missing tvars define
tvars = tf.trainable_variables()
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))

def discount_reward(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")
loglik = tf.log(input_y *(input_y - probability) + (1-input_y)*(input_y + probability))
loss = -tf.reduce_mean(loglik * advantages)

tvars = tf.trainable_variables()
newGrads = tf.gradients(loss, tvars)

xs, ys, drs = [], [], []
reward_sum = 0
episode_number = 1
total_episodes = 10000

with tf.Session() as sess:
    rendering = False
    init = tf.global_variables_initializer()
    sess.run(init)
    observation = env.reset()

    # why return value is gradBuff
    gradBuffer = sess.run(tvars)
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0


    while episode_number <= total_episodes:
        if reward_sum/batch_size > 100 or rendering == True:
            env.render()
            rendering = True

        x = np.reshape(observation, [1, D])
        tfprob = sess.run(probability, feed_dict={observations: x})
        action = 1 if np.random.uniform() < tfprob else 0
        xs.append(x)
        y = 1 - action
        ys.append(y)

        observation, reward, done, info = env.step(action)
        reward_sum += reward
        drs.append(reward)

        if done:
            episode_number += 1
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            xs, ys, drs = [], [], []

            discounted_err = discount_rewards(epr)
            discounted_err = 1 / discount_reward(epr)






