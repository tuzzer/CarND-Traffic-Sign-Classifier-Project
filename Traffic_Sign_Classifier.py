# Load pickled data
import pickle
import os

data_folder = os.path.join(os.getcwd(), "traffic-signs-data")
training_file = os.path.join(data_folder, "train.p")
validation_file = os.path.join(data_folder, "valid.p")
testing_file = os.path.join(data_folder, "test.p")

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train_raw, y_train = train['features'], train['labels']
X_valid_raw, y_valid = valid['features'], valid['labels']
X_test_raw, y_test = test['features'], test['labels']

### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

# Number of training examples
n_train = len(y_train)

# Number of validation examples
n_validation = len(y_valid)

# Number of testing examples.
n_test = len(y_test)

# What's the shape of an traffic sign image?
image_shape = X_train_raw[0].shape

# How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import csv

# Visualizations will be shown in the notebook.
y_train_by_class = [0]*n_classes
for sample in y_train:
    y_train_by_class[sample] += 1

# Extract the label text
class_labels_text = list(range(n_classes))
with open("signnames.csv", 'r') as csvfile:
    signnames_reader = csv.reader(csvfile, delimiter=',')
    # skip the header
    next(signnames_reader)
    for row in signnames_reader:
        class_labels_text[int(row[0])] = str(row[1])

# Sorting the from most to least
y_train_by_class = list(zip(class_labels_text, y_train_by_class))
sorted_y_train_by_class = sorted(y_train_by_class, key=lambda x: x[1], reverse=True)
sorted_y_train_by_class = list(zip(*sorted_y_train_by_class))

# Plot the graph
plt.figure(figsize=(15,4))
plt.bar(range(len(sorted_y_train_by_class[1])), sorted_y_train_by_class[1], 0.5)
plt.xlabel("Labels")
plt.ylabel("Training Sample Counts")
plt.title("Number of training samples by class")
plt.xticks(range(len(sorted_y_train_by_class[0])), sorted_y_train_by_class[0], rotation="vertical")
plt.show()

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include
### converting to grayscale, etc.
import numpy as np

# Greyscale the images
X_train = np.mean(X_train_raw, axis=3)
X_train = X_train.reshape(X_train.shape + (1,))
X_valid = np.mean(X_valid_raw, axis=3).squeeze()
X_valid = X_valid.reshape(X_valid.shape + (1,))
X_test = np.mean(X_test_raw, axis=3).squeeze()
X_test = X_test.reshape(X_test.shape + (1,))

image_shape = X_train[0].shape

from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)

### Define your architecture here.
import tensorflow as tf

EPOCHS = 20
BATCH_SIZE = 512

from tensorflow.contrib.layers import flatten


def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)

    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1 = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2 = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits


### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected,
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

x = tf.placeholder(tf.float32, (None,) + image_shape)
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):

        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './traffic-sign-lenet')
    print("Model saved")


def predict(X_data):
    sess = tf.get_default_session()
    predictions = sess.run(logits, feed_dict={x: X_data})
    return np.argmax(predictions, axis=1)

random_image_ids = np.random.choice(n_test, 2)

with tf.Session(config=config) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    predictions = predict(X_test_raw[random_image_ids])
    for i in range(len(random_image_ids)):
        x_sample = X_test_raw[random_image_ids[i]]
        plt.figure(figsize=(4, 4))
        plt.imshow(x_sample)
        plt.show()
        print("Prediction = %s" % class_labels_text[predictions[i]])

with tf.Session(config=config) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))