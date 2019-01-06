from mnist_loader import load_data_wrapper_cnn
import network2

training_data, validation_data, test_data = load_data_wrapper_cnn()
training_data = list(training_data)
validation_data = list(validation_data)

# import matplotlib.pyplot as plt
# plt.imshow(training_data[777][0].reshape((64,64)), cmap='Greys_r')
# plt.show()
# exit(0)
# import network
# net = network.Network([784, 30, 10])
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

#
# net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
# net.SGD(training_data, 30, 10, 0.1,
#         lmbda=5.0,
#         evaluation_data=validation_data,
#         monitor_training_accuracy=True,
#         monitor_training_cost=True,
#         monitor_evaluation_cost=True,
#         monitor_evaluation_accuracy=True)

from network4 import *
import matplotlib.pyplot as plt

batch_size = 20
EPOCH = 12
net = TestNetwork(
    [CNNLayer(1, 6, 3, ReLU, padding=1),
     MaxPoolingLayer(),
     CNNLayer(6, 16, 3, ReLU, padding=1, to_fc=True),
     MaxPoolingLayer(),
     ToFC(),
     FullyConnectedLayer(784, 120, ReLU),
     FullyConnectedLayer(120, 10, Sigmoid)],
    batch_size)
opt = SGD(net.params, 0.5, 0.1)
batchs = [training_data[i:i + batch_size] for i in range(0, len(training_data), batch_size)]
validate_batch = [validation_data[i:i + batch_size] for i in range(0, len(validation_data), batch_size)]
for epoch in range(EPOCH):
    for i, batch in enumerate(batchs[:20]):
        opt.zero_grad()
        net(batch)
        opt.step()
        print(i)
    correct = 0
    total_loss = 0
    for batch in validate_batch[:5]:
        x = np.array([i[0] for i in batch])
        y = np.array([i[1] for i in batch])
        output = np.argmax(net(batch)[0], axis=1).reshape(-1)
        correct += np.sum(output == y)
    print(correct)
    # print('Epoch {} finished, accuracy: {}/{}, loss is {}.'.format(epoch, correct, len(validation_data), total_loss))
