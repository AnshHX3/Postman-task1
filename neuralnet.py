import random
import numpy as np


def sigmoid(n):
    return 1 / (1 + np.exp(-n))


# Loss function: 0.5 * (output - target)^2
def loss_func(output, target):
    return 0.5 * ((output - target) ** 2)


# Derivative of Loss w.r.t Output: (output - target)
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


# -------------------------------------------------------------
# 1. INITIALIZATION (Fix: Generate unique weights per neuron)
# -------------------------------------------------------------
network = []
prev_inputs = len(raw_input)

for i in num_neuron:
    layer = []

    for _ in range(i):
        # Generate a distinct list of weights for EACH neuron
        wt = []
        for _ in range(prev_inputs):
            random_wt = round(random.uniform(-1, 1), 2)
            wt.append(random_wt)

        neur = neuron(weight=wt, bias=0.1)
        layer.append(neur)

    network.append(layer)
    prev_inputs = i

# -------------------------------------------------------------
# TRAINING LOOP (Minimal Addition: Runs Forward & Backward 5 times)
# -------------------------------------------------------------
epochs = 10
learning_rate = 0.8  # Slightly higher rate so changes are visible in 5 steps

for epoch in range(1, epochs + 1):
    # Reset layer inputs and current inputs at start of every pass
    all_layer_inputs = [raw_input]
    current_inputs = raw_input

    # -------------------------------------------------------------
    # 2. FORWARD PASS
    # -------------------------------------------------------------
    for layer in network:
        layer_outputs = []
        for neur in layer:
            out = neur.fwdpass(current_inputs)
            layer_outputs.append(out)
        all_layer_inputs.append(layer_outputs)
        current_inputs = layer_outputs

    # -------------------------------------------------------------
    # 3. LOSS & PRINT PROGRESS
    # -------------------------------------------------------------
    final_output = current_inputs[0]
    start_loss = loss_func(final_output, target)

    print(f"Epoch {epoch} | Output: {final_output:.6f} | Loss: {start_loss:.6f}")

    # -------------------------------------------------------------
    # 4. BACKWARD PASS & PARAMETER UPDATES
    # -------------------------------------------------------------
    next_layer_deltas = []  # Stores error deltas from the layer above
    next_layer_neurons = []  # Stores neuron objects from the layer above

    # Loop backwards using indices: 2 (Output Layer), 1 (Hidden 2), 0 (Hidden 1)
    for layer_idx in range(len(network) - 1, -1, -1):
        current_layer = network[layer_idx]
        current_inputs_to_layer = all_layer_inputs[layer_idx]
        current_layer_deltas = []

        for neur_idx in range(len(current_layer)):
            neur = current_layer[neur_idx]
            a = neur.activated_output
            da_dz = a * (1.0 - a)  # Sigmoid derivative

            # ---------------------------------------------------------
            # A. CALCULATE DELTA (Error signal)
            # ---------------------------------------------------------
            if layer_idx == len(network) - 1:
                # Output Layer: dL/da = (output - target)
                dL_da = neur.activated_output - target
                delta = dL_da * da_dz
            else:
                # Hidden Layers: dL/da = sum(next_delta * connecting_weight)
                dL_da = 0
                for next_idx in range(len(next_layer_neurons)):
                    next_neur = next_layer_neurons[next_idx]
                    next_delta = next_layer_deltas[next_idx]
                    # Weight connecting this neuron to the next layer's neuron
                    connecting_wt = next_neur.weight[neur_idx]
                    dL_da += next_delta * connecting_wt

                delta = dL_da * da_dz

            current_layer_deltas.append(delta)

            # ---------------------------------------------------------
            # B. UPDATE WEIGHTS AND BIAS (Gradient Descent)
            # ---------------------------------------------------------
            for w_idx in range(len(neur.weight)):
                grad_w = delta * current_inputs_to_layer[w_idx]
                neur.weight[w_idx] -= learning_rate * grad_w

            grad_b = delta * 1.0
            neur.bias -= learning_rate * grad_b

        # Pass current layer's deltas & neurons up to the preceding layer
        next_layer_deltas = current_layer_deltas
        next_layer_neurons = current_layer