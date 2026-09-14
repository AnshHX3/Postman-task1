import random
import numpy as np


def sigmoid(n):
    return 1 / (1 + np.exp(-n))


def loss_func(output, target):
    return 0.5 * ((output - target) ** 2)


def derivative_func(output, target):
    return output - target


num_neuron = [3, 2, 1]
raw_input = [0.2, 0.3, 0.7]
target = 1.0


class neuron:

    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias
        self.z = 0
        self.activated_output = 0

    def fwdpass(self, current_inputs):
        weighted_sum = 0
        for i in range(len(current_inputs)):
            weighted_sum += self.weight[i] * current_inputs[i]

        self.z = weighted_sum + self.bias
        self.activated_output = sigmoid(self.z)
        return self.activated_output

network = []
prev_inputs = len(raw_input)

for i in num_neuron:
    layer = []

    for _ in range(i):
        wt = []
        for _ in range(prev_inputs):
            random_wt = round(random.uniform(-1, 1), 2)
            wt.append(random_wt)

        neur = neuron(weight=wt, bias=0.1)
        layer.append(neur)

    network.append(layer)
    prev_inputs = i


epochs = 10
learning_rate = 0.8  

for epoch in range(1, epochs + 1):
    all_layer_inputs = [raw_input]
    current_inputs = raw_input

    for layer in network:
        layer_outputs = []
        for neur in layer:
            out = neur.fwdpass(current_inputs)
            layer_outputs.append(out)
        all_layer_inputs.append(layer_outputs)
        current_inputs = layer_outputs

    final_output = current_inputs[0]
    start_loss = loss_func(final_output, target)

    print(f"Epoch {epoch} | Output: {final_output:.6f} | Loss: {start_loss:.6f}")

    next_layer_deltas = []  
    next_layer_neurons = []  

    for layer_idx in range(len(network) - 1, -1, -1):
        current_layer = network[layer_idx]
        current_inputs_to_layer = all_layer_inputs[layer_idx]
        current_layer_deltas = []

        for neur_idx in range(len(current_layer)):
            neur = current_layer[neur_idx]
            a = neur.activated_output
            da_dz = a * (1.0 - a)  

            if layer_idx == len(network) - 1:
                dL_da = neur.activated_output - target
                delta = dL_da * da_dz
            else:
                dL_da = 0
                for next_idx in range(len(next_layer_neurons)):
                    next_neur = next_layer_neurons[next_idx]
                    next_delta = next_layer_deltas[next_idx]

                    connecting_wt = next_neur.weight[neur_idx]
                    dL_da += next_delta * connecting_wt

                delta = dL_da * da_dz

            current_layer_deltas.append(delta)

            for w_idx in range(len(neur.weight)):
                grad_w = delta * current_inputs_to_layer[w_idx]
                neur.weight[w_idx] -= learning_rate * grad_w

            grad_b = delta * 1.0
            neur.bias -= learning_rate * grad_b


        next_layer_deltas = current_layer_deltas
        next_layer_neurons = current_layer