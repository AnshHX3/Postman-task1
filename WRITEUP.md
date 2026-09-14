I have built a 3 layer neural network - this can be extended to multiple layers - shown by the length of the list 'num_neuron' in line 17. 
Each value in the list is the number of neurons in the 1st, 2nd ... layers.
The raw inputs for each neuron is given by the list 'raw_input' in line 18. The values can be edited and the output changes accordingly.
The activation function of my choice is the sigmoid function, sig(n) = 1/(1+e^(-n))
I created a class Neuron, which can have several properties i.e. weight, bias, and an activated output.
The structure of the neural network itself is given by the code in line 65 to line 71
The main backpropagation code is there in line 81 to line 112
All of the above code , the forward and backward pass , is put into a for loop of epochs to cycle through and check the changes in the outputs.
I have set the number of epochs to 10 - line 58
I have set the learning rate to 0.8. Increasing this would result in larger changes to the outputs, and decreasing this, vice versa.
