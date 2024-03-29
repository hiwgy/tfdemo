
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.contrib.eager as tfe

# If use eager, should call this first
tf.enable_eager_execution()

class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.W = tfe.Variable(5., name='weight')
        self.B = tfe.Variable(10., name='bias')
    def predict(self, inputs):
        return inputs * self.W + self.B

# A toy dataset of points around 3*x+2
NUM_EXAMPLES = 1000
training_inputs = tf.random_normal([NUM_EXAMPLES])
noise = tf.random_normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise

# A loss function using mean-squared error
def loss(model, inputs, targets):
    error = model.predict(inputs) - targets
    return tf.reduce_mean(tf.square(error))

# Return the derivative of loss with respect tot weight and bias
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, [model.W, model.B])

model = Model()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

print("Initial loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))

# Trainning loop
for i in range(300):
    grads = grad(model, training_inputs, training_outputs)
    optimizer.apply_gradients(zip(grads, [model.W, model.B]),
                              global_step=tf.train.get_or_create_global_step())

    if i % 20 == 0:
        print("Loss at step {:03d}: {:.3f}".format(i, loss(model, training_inputs, training_outputs)))

print("Final loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))
print("W = {:.3f}, B = {:.3f}".format(model.W.numpy(), model.B.numpy()))

