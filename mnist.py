import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
learning_rate = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

X = tf.placeholder(tf.float32, [None, 784])  # 초기 입력값
Y = tf.placeholder(tf.float32, [None, 10])  # 판단 해야할 결과 개수

keep_prob = tf.placeholder(tf.float32)

W1 = tf.get_variable("W1", shape=[784, 512],  # 784개 인풋이 들어가고 512개의 아웃풋
                     initializer=tf.contrib.layers.xavier_initializer())
# 초기화를 xavier 으로 하면 초기의 initializer 가 아주 잘 되어 나온다.
b1 = tf.Variable(tf.random_normal([512]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
# layer 층이 많아지면서 overffiting 이 일어나는걸 방지한다.
# doropout 전체중의 몇프로의 네트워크를 끊어줄 것인지 결정한다 통상적으로 0.5~0.7 정도 끊어준다.
# 테스팅 할때는 네트워크를 총 동원해야하기 때문에 1으로 설정한다

W2 = tf.get_variable("W2", shape=[512, 512],  # 512개 인풋이 들어가고 512개의 아웃풋
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
# layer 층이 많아지면서 overffiting 이 일어나는걸 방지한다.

W3 = tf.get_variable("W3", shape=[512, 512],  # 512개 인풋이 들어가고 512개의 아웃풋
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
# layer 층이 많아지면서 overffiting 이 일어나는걸 방지한다.

W4 = tf.get_variable("W4", shape=[512, 512],  # 512개 인풋이 들어가고 512개의 아웃풋
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([512]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
# layer 층이 많아지면서 overffiting 이 일어나는걸 방지한다.

W5 = tf.get_variable("W5", shape=[512, 10],  # 512개 인풋이 들어가고 10개의 아웃풋
                     initializer=tf.contrib.layers.xavier_initializer())
# 마지막 레이어는 10개가 나와야하는건 고정
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L4, W5) + b5

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# AdamOptimizer 다양한 Optimizer 가 있지만 그중에 Adam Optimizer 가 성능이 가장 좋다
# GradientDescentOptimizer 를 사용했을때 보다 성능이 더 높게 나옴

sess = tf.Session()
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter('./logs', sess.graph)
for epoch in range(training_epochs):  # epoch 만큼 수행
    avg_cost = 0

    for i in range(total_batch):  # 한번에 학습할 데이터 량
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))
# 테스팅 케이스 시작할때는 keep_prob 을 1 으로 설정

r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1}))


# https://github.com/hunkim/DeepLearningZeroToAll 참조
