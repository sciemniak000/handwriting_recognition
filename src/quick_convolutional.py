from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import code
import imageResize as ir
import numpy as np

#download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#reshape data to fit model
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

#one-hot encode target column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#create model
model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(28, 28, 1)))
model.add(Conv2D(32, kernel_size=3, activation="relu"))
model.add(Flatten())
model.add(Dense(10, activation="softmax"))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

#predict first 4 images in the test set
model.predict(X_test[:4])

# 1
image1 = np.reshape(ir.resizeToMNISTFormat(imageFile=r"..\real_test_images\IMG_20190612_062613.jpg"), (1, 28, 28, 1))

# 2
image2 = np.reshape(ir.resizeToMNISTFormat(imageFile=r"..\real_test_images\IMG_20190612_062603.jpg"), (1, 28, 28, 1))

# 3
image3 = np.reshape(ir.resizeToMNISTFormat(imageFile=r"..\real_test_images\IMG_20190612_062617.jpg"), (1, 28, 28, 1))

# 7
image4 = np.reshape(ir.resizeToMNISTFormat(imageFile=r"..\real_test_images\IMG_20190612_061719.jpg"), (1, 28, 28, 1))

# 2
image5 = np.reshape(ir.resizeToMNISTFormat(imageFile=r"..\real_test_images\IMG_20190612_085511.jpg"), (1, 28, 28, 1))

# 3
image6 = np.reshape(ir.resizeToMNISTFormat(imageFile=r"..\real_test_images\IMG_20190612_085527.jpg"), (1, 28, 28, 1))

# 4
image7 = np.reshape(ir.resizeToMNISTFormat(imageFile=r"..\real_test_images\IMG_20190612_085543.jpg"), (1, 28, 28, 1))

# 5
image8 = np.reshape(ir.resizeToMNISTFormat(imageFile=r"..\real_test_images\IMG_20190612_085549.jpg"), (1, 28, 28, 1))

# 6
image9 = np.reshape(ir.resizeToMNISTFormat(imageFile=r"..\real_test_images\IMG_20190612_085553.jpg"), (1, 28, 28, 1))

# 7
image10 = np.reshape(ir.resizeToMNISTFormat(imageFile=r"..\real_test_images\IMG_20190612_085557.jpg"), (1, 28, 28, 1))

code.interact(local=locals())