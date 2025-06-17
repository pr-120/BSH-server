import numpy as np


class ModelOptimized(object):
    def __init__(self, learn_rate, num_configs):
        # Initialize hyperparams
        self.learn_rate = learn_rate

        # Initialize action set
        self.allowed_actions = np.asarray(range(num_configs))

    def forward(self, weights1, weights2, bias_weights1, bias_weights2, epsilon, inputs):
        # print("MODEL: inputs", inputs.shape, inputs)
        # print("MODEL: inputs min/max", inputs.shape, np.min(inputs), np.argmin(inputs), np.max(inputs), np.argmax(inputs))

        # ==============================
        # Q-VALUES
        # ==============================

        # Forward pass through neural network to compute Q-values
        # print("MODEL: w1", weights1.shape, weights1)
        # print("MODEL: w1 min/max", weights1.shape, np.min(weights1), np.argmin(weights1), np.max(weights1), np.argmax(weights1))
        # print("MODEL: bw1 min/max", bias_weights1.shape, np.min(bias_weights1), np.argmin(bias_weights1), np.max(bias_weights1), np.argmax(bias_weights1))
        adaline1 = np.dot(weights1.T, inputs) + bias_weights1
        # print("MODEL: ad1 dot", np.dot(weights1.T, inputs))
        # print("MODEL: ad1 min/max", adaline1.shape, np.min(adaline1), np.argmin(adaline1), np.max(adaline1), np.argmax(adaline1))

        hidden1 = 1 / (1 + np.exp(np.clip(-adaline1, -500, 500)))  # logistic activation
        # hidden1 = adaline1 * (adaline1 > 0)  # ReLU activation, x if a > 0 else 0
        # hidden1 = adaline1 / (1 + np.exp(-adaline1))  # SiLU activation, x*sig(x) = x/(1+e^-x)

        # print("MODEL: hidden1 min/max", hidden1.shape, np.min(hidden1), np.argmin(hidden1), np.max(hidden1), np.argmax(hidden1))
        # print("MODEL: w2", weights2.shape, weights2)
        # print("MODEL: w2 min/max", weights2.shape, np.min(weights2), np.argmin(weights2), np.max(weights2), np.argmax(weights2))
        # print("MODEL: bw2 min/max", bias_weights2.shape, np.min(bias_weights2), np.argmin(bias_weights2), np.max(bias_weights2), np.argmax(bias_weights2))

        adaline2 = np.dot(weights2.T, hidden1) + bias_weights2
        # print("MODEL: ad2 dot", np.dot(weights2.T, hidden1))
        # print("MODEL: ad2 min/max", adaline2.shape, np.min(adaline2), np.argmin(adaline2), np.max(adaline2), np.argmax(adaline2))

        # q = 1 / (1 + np.exp(-adaline2))  # h2, logistic activation
        # q = adaline2 * (adaline2 > 0)  # h2, ReLU activation, x if a > 0 else 0
        q = adaline2 / (1 + np.exp(np.clip(-adaline2, -500, 500)))  # h2, SiLU activation, x*sig(x) = x/(1+e^-x)

        # print("MODEL: Q", q.shape, "\n", q)

        # ==============================
        # POLICY
        # ==============================

        # Choose action based on epsilon-greedy policy
        possible_a = self.allowed_actions  # technically an array of indexes
        q_a = q[possible_a]

        if np.random.random() < epsilon:  # explore randomly
            sel_a = possible_a[np.random.randint(possible_a.size)]
            # print("MODEL: random action", sel_a)
        else:  # exploit greedily

            # TODO: for some reason the q values converge from negative to zero when they are supposed to be
            #       punished (-reward), ex: [[-2.4561842563026063e-21], [-8.589077436665736e-12], [-0.00013407748641937532], [-0.021268360634877838], [-8.892077069642031e-125], [-6.096650285262963e-69]]
            #       (In the exaple one can see that the most punished configs (4 and 5) are the closest to zero)
            #       This means that when all values are zero at the same time, the worst configurations are chosen as
            #       they are the largest values. Current hack uses np.argmin when all values are negative.
            #       Find and fix actual problem.
            if np.all(q_a < 0):
                argmax = np.argmin(q_a)
            else:
                argmax = np.argmax(q_a)
            # print("MODEL: argmax", argmax, "of", q_a, "for", possible_a)
            sel_a = possible_a[argmax]
            # print("MODEL: greedy action", sel_a)

        return hidden1, q, sel_a

    def backward(self, q, q_err, hidden, weights1, weights2, bias_weights1, bias_weights2, inputs):
        # Backpropagation of error through neural network

        # ==============================
        # COMPUTE DELTA
        # ==============================

        # print("MODEL back: inputs err", inputs.shape, q_err.shape, inputs.T, q_err.T, sep="\n")

        delta2 = 1 / (1 + np.exp(-q)) * (
                1 + q * (1 - (1 / (1 + np.exp(-q))))) * q_err  # derivative SiLU: sig(x) * (1 + x(1 - sig(x)))
        # delta2 = (q > 0) * q_err  # derivative ReLU: 1 if x > 0 else 0
        # delta2 = q * (1 - q) * q_err  # derivative logistic: f(x) * (1 - f(x))

        delta_weights2 = np.outer(hidden, delta2.T)
        # print("MODEL back: d2", delta2.shape, delta2)
        # print("MODEL back: d2 min/max", delta2.shape, np.min(delta2), np.argmin(delta2), np.max(delta2), np.argmax(delta2))
        # print("MODEL back: dw2", delta_weights2.shape, delta_weights2)
        # print("MODEL back: dw2 min/max", delta_weights2.shape, np.min(delta_weights2), np.argmin(delta_weights2), np.max(delta_weights2), np.argmax(delta_weights2))

        # delta1 = 1 / (1 + np.exp(-hidden)) * (1 + hidden*(1 - (1 / (1 + np.exp(-hidden))))) * np.dot(weights2, delta2)  # derivative SiLU: sig(x) * (1 + x(1 - sig(x)))
        # delta1 = (hidden > 0) * np.dot(weights2, delta2)  # derivative ReLU: 1 if q > 0 else 0
        delta1 = hidden * (1 - hidden) * np.dot(weights2, delta2)  # derivative logistic: f(x) * (1 - f(x))

        delta_weights1 = np.outer(inputs, delta1)
        # print("MODEL back: d1", delta1.shape, delta1)
        # print("MODEL back: d1 min/max", delta1.shape, np.min(delta1), np.argmin(delta1), np.max(delta1), np.argmax(delta1))
        # print("MODEL back: dw1", delta_weights1.shape, delta_weights1)
        # print("MODEL back: dw1 min/max", delta_weights1.shape, np.min(delta_weights1), np.argmin(delta_weights1), np.max(delta_weights1), np.argmax(delta_weights1))

        # ==============================
        # UPDATE WEIGHTS
        # ==============================

        # print("MODEL: weights1 before", weights1.shape, np.min(weights1), np.argmin(weights1), np.max(weights1), np.argmax(weights1))
        # print("MODEL: weights2 before", weights2.shape, np.min(weights2), np.argmin(weights2), np.max(weights2), np.argmax(weights2))

        weights1 += self.learn_rate * delta_weights1
        weights2 += self.learn_rate * delta_weights2
        bias_weights1 += self.learn_rate * delta1
        bias_weights2 += self.learn_rate * delta2

        # print("MODEL: weights1 after", weights1.shape, np.min(weights1), np.argmin(weights1), np.max(weights1), np.argmax(weights1))
        # print("MODEL: weights2 after", weights2.shape, np.min(weights2), np.argmin(weights2), np.max(weights2), np.argmax(weights2))
        # print("MODEL: bias1 after", bias_weights1.shape, np.min(bias_weights1), np.argmin(bias_weights1), np.max(bias_weights1), np.argmax(bias_weights1))
        # print("MODEL: bias2 after", bias_weights2.shape, np.min(bias_weights2), np.argmin(bias_weights2), np.max(bias_weights2), np.argmax(bias_weights2))

        return weights1, weights2, bias_weights1, bias_weights2
