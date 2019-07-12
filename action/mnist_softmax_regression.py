
import tensorflow as tf
import dataset

sess = tf.InteractiveSession()

mnist_train = dataset.train("./mnist_data");
mnist_test = dataset.test("./mnist_data");

# https://www.tensorflow.org/guide/datasets
print(mnist_train.output_shapes, mnist_train.output_types)
print(mnist_test.output_shapes, mnist_test.output_types)
batched_train = mnist_train.batch(100)

iterator = batched_train.make_one_shot_iterator()
next_element = iterator.get_next();

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# y is a matrix
y = tf.nn.softmax(tf.matmul(x, W) + b)

# TODO: try [None, 10]
# y_ = tf.placeholder(tf.int32, [None])
y_ = tf.placeholder(tf.float32, [None, 10])

# cause y is a 2D matrix, so the predicted answer lay on the second axis
# cross_entropy = tf.reduce_mean(-tf.gather(tf.log(y), y_, axis=1))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
tf.global_variables_initializer().run()

for i in range(50):
    batch_xs, raw_batch_ys = sess.run(next_element)
    # transform the format of the label
    batch_ys = sess.run(tf.one_hot(raw_batch_ys, 10))
    sess.run([train_step], feed_dict={x: batch_xs, y_: batch_ys})

# now test the accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

batched_test = mnist_test.batch(1000)
test_iterator = batched_test.make_one_shot_iterator()
test_next_element = test_iterator.get_next()
test_xs, raw_test_ys = sess.run(test_next_element)
test_ys = sess.run(tf.one_hot(raw_test_ys, 10))
print(accuracy.eval({x: test_xs, y_: test_ys}))

