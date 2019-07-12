
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
y_ = tf.placeholder(tf.int32, [None])
# TODO: why this not work?
# y_ = tf.placeholder(tf.int32, [None, 1]) 
# y_ = tf.placeholder(tf.float32, [None, 10])

# cause y is a 2D matrix, note the meaning of reduce_sum
one_hot_y = tf.one_hot(y_, 10)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.log(y) * one_hot_y, axis=1))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
tf.global_variables_initializer().run()

for i in range(50):
    batch_xs, batch_ys = sess.run(next_element)
    sess.run([train_step], feed_dict={x: batch_xs, y_: batch_ys})

# now test the accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.cast(y_, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

batched_test = mnist_test.batch(1000)
test_iterator = batched_test.make_one_shot_iterator()
test_next_element = test_iterator.get_next()
test_xs, test_ys = sess.run(test_next_element)
print(accuracy.eval({x: test_xs, y_: test_ys}))

