from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
import numpy as np

with open('..\\data\\train-images-idx3-ubyte.gz', 'rb') as f:
    train_images = extract_images(f)
with open('..\\data\\train-labels-idx1-ubyte.gz', 'rb') as f:
    train_lab = extract_labels(f)

with open('..\\data\\t10k-images-idx3-ubyte.gz', 'rb') as f:
    test_images = extract_images(f)
with open('..\\data\\t10k-labels-idx1-ubyte.gz', 'rb') as f:
    test_lab = extract_labels(f)

# train_images = np.zeros(train_images.shape) + train_images
train_l = np.array(train_images)


train_l = np.delete(train_l, (train_l.shape[1] - 1), axis=1)
train_l = np.concatenate((np.zeros((train_l.shape[0], 1, train_l.shape[2], train_l.shape[3])), train_l), axis=1)

train_r = np.array(train_images)
train_r = np.delete(train_r, 0, axis=1)
train_r = np.concatenate((train_r, np.zeros((train_r.shape[0], 1, train_r.shape[2], train_r.shape[3]))), axis=1)

train_u = np.array(train_images)
train_u = np.delete(train_u, train_u.shape[2] - 1, axis=2)
train_u = np.concatenate((np.zeros((train_u.shape[0], train_u.shape[1], 1, train_u.shape[3])), train_u), axis=2)


train_d = np.array(train_images)
train_d = np.delete(train_d, 0, axis=2)
train_d = np.concatenate((train_d, np.zeros((train_d.shape[0], train_d.shape[1], 1, train_d.shape[3]))), axis=2)

train_images = np.concatenate((train_images, train_l, train_r, train_u, train_d))

# train_images = train_images.reshape((train_images.shape[0], train_images.shape[1] * train_images.shape[2], 1))
# test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2], 1)

train_labels = np.zeros((train_lab.shape[0], 10))
test_labels = np.zeros((test_lab.shape[0], 10))

for i in range(0, test_lab.shape[0]):
    temp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    temp[test_lab[i]] = 1.
    test_labels[i] += temp

for i in range(0, train_lab.shape[0]):
    temp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    temp[train_lab[i]] = 1.
    train_labels[i] += temp

# train_labels = np.reshape(train_labels, (train_labels.shape[0], 10, 1))
train_labels = np.concatenate((train_labels, train_labels, train_labels, train_labels, train_labels))

# test_labels = np.reshape(test_labels, (test_labels.shape[0], 10, 1))

del test_lab
del train_lab
