## Neural Network Machine Learning Structure:

The Neural Network structure is by far the most popular and aims to mimic the process of how neurons work in the brain. These neurons receive various inputs that are then transformed before being sent to the next neuron. These neurons can then be "trained" to perform the correct transformations to provide the correct final answer.

The neural network has three main layers:

* **Input layer**: This is the first layer of nodes in the neural network. These nodes each receive a single data input that is then passed on to the hidden layer. This means that the number of nodes in this layer always matches the network's number of inputs (or data parameters).
* **Output layer**: This is the last layer of nodes in the neural network. These nodes send the output from the network once it has been received from the hidden layer. Therefore, the number of nodes in this layer will always be the same as the network's number of outputs.
* **Hidden layer**: This is the layer of nodes between the neural network's input and output layers. With a simple neural network, this will only be one layer of nodes. However, for additional learning opportunities, we could add more layers to create a deep neural network. This layer is where the neural network's main action takes place. Each node within the neural network's hidden layer receives multiple inputs from the nodes in the previous layer and will then transmit their answers to multiple nodes in the next layer.

## Neural Network Training:

There are two steps to training the network: the [feed-forward] step and the [back-propagation] step.

_Note:_ Inputs are not directly added. Instead, they are multiplied by a weight value first. This helps the neural network decide which inputs should contribute more to the output than others.
The addition's output is not directly transmitted out. Instead, the output is first entered into what is called an activation function. In essence, this decides if the neuron (node) will be active or not. It does this by ensuring that the output, no matter the input, will always be a decimal between 0 and 1 (or between −1 and 1).

### Feed-Forward Loop:

The feed-forward loop is how we send data through the network and get an answer on the other side. Once our network has been trained, this is the only step we perform. At this point, we stop training and simply want an answer from the network. To complete one round of the feed-forward step, we have to perform the following:

* Normalise all of the inputs: To allow our neural network to decide which inputs are most important in helping it to decide the answer, we need to normalise them. As mentioned before, each node in the network tries to keep its answer between 0 and 1. If we have one input with a range of 0 to 50 and another with a range of 0 to 2, our network won't be able to properly consume the input. Therefore, we normalise the inputs first by adjusting them so that their ranges are all the same. In our example here, we would take the inputs with a 0 to 50 range and divide all of them by 25 to change their ranges to 0 to 2.
* Feed the inputs to our nodes in the input layer: Once normalised, we can provide one data entry for each input node in our network.
* Propagate the data through the network: At each node, we add all the inputs and run them through the activation function to get the node's output. This output then becomes the input for the next layer of nodes. We repeat this process until we get to our network's output layer.
* Read the output from the network: At the output layer of the network, we receive the output from our nodes. The answer will be a decimal between 0 and 1, but, for decision-making, we'll round it to get a binary answer from each output node.

### Back-Propagation Loop:

When we are training our network, the feed-forward loop is only half of the process. Once we receive the answers from our network, we need to tell it how close it was to the correct answer. This is the back-propagation step. Here, we perform the following steps:

* Calculate the difference in received outputs vs expected outputs: As mentioned before, the activation function will provide a decimal answer between 0 and 1. Since we know that the answer has to be either 0 or 1, we can calculate the difference in the answer. This difference tells us how close the neural network was to the correct answer.
* Update the weights of the nodes: Using the difference calculated in the previous step, we can start to update the weights of each input to the nodes in the output layer. We won't dive too deep into this update process, as it often involves a bit of complex maths to decide what update should be made.
* Propagate the difference back to the other layers: This is where the term back-propagation comes from. Once the weights of the nodes in the output layer have been updated, we can calculate what the difference would be for the previous nodes. Once again, this difference is then used to update the weights of the nodes in that layer before being propagated backwards even more. We continue this process of back-propagation until the weights for the input layer have been updated.




Source: [TryHackMe Advent Of Cyber 2023](https://tryhackme.com/room/adventofcyber2023)

Ref: Task 20 [Day 14] Machine Learning

