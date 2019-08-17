import ffnn
# import loadMNIST
# import convolution as cv
import imageResize as ir
import matplotlib.pyplot as plt
import code

net = ffnn.Network([784, 100, 10], r"..\ready_networks\8276$30_10__0.1")

# loadMNIST.train_images = loadMNIST.np.reshape(loadMNIST.train_images, (loadMNIST.train_images.shape[0],
#                                                                        784, 1))
# loadMNIST.train_labels = loadMNIST.np.reshape(loadMNIST.train_labels, (loadMNIST.train_labels.shape[0],
#                                                                        10, 1))
#
#
# loadMNIST.test_images = loadMNIST.np.reshape(loadMNIST.test_images, (loadMNIST.test_images.shape[0],
#                                                                      784, 1))
# loadMNIST.test_labels = loadMNIST.np.reshape(loadMNIST.test_labels, (loadMNIST.test_labels.shape[0],
#                                                                   10, 1))
# net = ffnn.Network([784, 100, 10])

# for i in range(5):
#     net = ffnn.Network([784, 100, 10])
#     net.train_network(list(zip(loadMNIST.train_images, loadMNIST.train_labels)), 30, 10, 0.1,
#                   list(zip(loadMNIST.test_images, loadMNIST.test_labels)))

# print(net.evaluate(list(zip(loadMNIST.test_images, loadMNIST.test_labels))))

image1 = ir.resizeToMNISTFormat(imageFile=r"..\real_test_images\IMG_20190612_062613.jpg")
image2 = ir.resizeToMNISTFormat(imageFile=r"..\real_test_images\IMG_20190612_062603.jpg")
image3 = ir.resizeToMNISTFormat(imageFile=r"..\real_test_images\IMG_20190612_062617.jpg")
image4 = ir.resizeToMNISTFormat(imageFile=r"..\real_test_images\IMG_20190612_061719.jpg")

image5 = ir.resizeToMNISTFormat(imageFile=r"..\real_test_images\IMG_20190612_085511.jpg")
image6 = ir.resizeToMNISTFormat(imageFile=r"..\real_test_images\IMG_20190612_085527.jpg")
image7 = ir.resizeToMNISTFormat(imageFile=r"..\real_test_images\IMG_20190612_085543.jpg")
image8 = ir.resizeToMNISTFormat(imageFile=r"..\real_test_images\IMG_20190612_085549.jpg")
image9 = ir.resizeToMNISTFormat(imageFile=r"..\real_test_images\IMG_20190612_085553.jpg")
image10 = ir.resizeToMNISTFormat(imageFile=r"..\real_test_images\IMG_20190612_085557.jpg")

def ans(imag):
    plt.imshow(imag, cmap="gray")
    plt.show()
    print(net.get_answer(net.feedforward(ffnn.np.reshape(
    imag, (784, 1)))))

code.interact(local=locals())

# test = cv.ConvolutionalNetwork([cv.ConvolutionLayer(20, (26, 26), (3, 3)), cv.PoolLayer(20, (25, 25), (2, 2), 1),
#                                 cv.OutputLayer(), cv.FullyConnectedLayer(100), cv.FullyConnectedLayer(10)])
#
# training_set = list(zip(loadMNIST.train_images, loadMNIST.train_labels))
# test_set = list(zip(loadMNIST.test_images, loadMNIST.test_labels))

# training_set = training_set[0:2000]
# test_set = test_set[0:1000]

# testImg = ir.resizeToMNISTFormat(file)

# print(cv.real_relu(loadMNIST.np.zeros((28, 28))).shape)
# test.determine_number(loadMNIST.test_images[0])
# test.test_the_net(0, test_set)
# test.train_network(training_set, 10, 10, 0.3, test_set)
