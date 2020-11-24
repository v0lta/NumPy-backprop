import numpy as np

class SGD:
    def __init__(self, lr):
        self.lr = lr

    def step(self, net, grads):
        param_keys = net.weights.keys()
        for weight_key in param_keys:
            cgrads = np.expand_dims(np.mean(grads[weight_key], 0), 0)
            net.weights[weight_key] += -self.lr*cgrads


class RMSprop(SGD):
    def __init__(self, lr=0.01, rho=0.99):
        super().__init__(lr)
        self.delta = 1e-6
        self.r = None
        self.rho = rho

    def step(self, net, grads):
        param_keys = net.weights.keys()

        if self.r is None:
            self.r = {}
            for weight_key in param_keys:
                grad = np.expand_dims(np.mean(grads[weight_key], 0), 0)
                self.r[weight_key] = np.zeros_like(grad)

        for weight_key in param_keys:
            cgrads = np.expand_dims(np.mean(grads[weight_key], 0), 0)
            self.r[weight_key] = self.rho*self.r[weight_key] \
                                + (1 - self.rho)*cgrads*cgrads
            step_size = -self.lr / (np.sqrt(self.delta + self.r[weight_key]))
            net.weights[weight_key] += step_size*cgrads