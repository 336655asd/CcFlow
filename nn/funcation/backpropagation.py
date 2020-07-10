import numpy as np

class NeuralNerwork():
    def __init__(self, sizes):
        self.sizes  = sizes
        self.num_layers = len(self.sizes)
        self.weights = [np.random.randn(y, x) for x,y in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(x) for x in sizes[1:]]

    def forward(self, inputs):
        activations = []
        for weight, bias in zip(self.weights, self.biases):
            inputs = self.sigmoid(np.dot(weight, inputs) + bias)
            activations.append(inputs)
        return inputs, activations

    def backforward(self):
        return

    def SGD(self, data_x, data_y, epochs, batch_size, lr):
        batchs = np.ceil(len(data_y)/batch_size)
        for epoch in epochs:
            for batch in batchs[:-1]:
                train_x = data_x[batch*batch_size:(batch+1)*batch_size]
                train_y = data_y[batch*batch_size:(batch+1)*batch_size]
                self.update_mini_batch(train_x, train_y, lr)
    
    def update_mini_batch(self, train_x, train_y, lr):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        
        outputs, activations = self.forward(train_x)
        deltas  = (outputs-train_x)(outputs*(1-outputs))

        for o,a,dt in zip(outputs, activations, deltas):
            nabla_w[-1] += np.dot(dt, o[-2].T)
            nabla_b[-1] += dt
            for layer in (2, self.num_layers):
                dt = np.dot(self.weights[-layer-1].T, dt)*a[-layer]
                nabla_w[-layer] += np.dot(dt, o[-layer+1].T)
                nabla_b[-layer] += dt

        nabla_w = [w.mean() for w in nabla_w]
        nabla_b = [b.mean() for b in nabla_b]
        self.weights = [w-lr*nw for w,nw in zip(self.weights, nabla_w)]
        self.biases = [b-lr*nb for b,nb in zip(self.biases, nabla_b)]

    def loss(self):
        return

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
