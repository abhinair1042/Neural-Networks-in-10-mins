# The Main Ideas Behind Neural Networks

## 1. About Weights

Neurons are essentially nodes that do simple multiplication and addition. A single neuron can take in an input, and apply a $Weight$ to it. The weight defines the strength of a connection and can be thought of as a multiplication factor that is applied to the input signal. So for an input 3.5 and for a connection to a neuron with a weight of 2, the output of the neuron is 7. The output of a neuron is called an $Activation$.

The weights are what is learned during training and what is loaded in pre-trained models.

An interesting perspective on neural networks: because they have a weight for each neuron, and add the fact that neurons also have the ability of adding a bias to the input, you can express their operation as $y = wx + b$. Essentially a linear equation. So each neuron contributes a specific amount to a certain output value. Finding out what that contribution is, is what training is all about.

## 2. Matrics

PyTorch, which is one of the frameworks used by ML Developers, is just a mathematical library to execute matrix operations and calculus. For a layer of horizontally stacked neurons that are connected to another layer of horizontally stacked neurons, you can express the operation as a matrix multiplication.

<p align="center">
  <img src="https://i.stack.imgur.com/8knTX.png" alt="drawing" width="600"/>
</p>

In the above figure, the $w_i,_j$ is basically the strength of the connection from neuron $x_i$ to neuron $y_j$. Arrange it into a matrix or a Tensor (a multi-dimensional array) and you can express the operation as a matrix multiplication.

```math
\begin{bmatrix}
    w_{1,1} & w_{1,2} & w_{1,3} \\\
    w_{2,1} & w_{2,2} & w_{2,3} \\\
    w_{3,1} & w_{3,2} & w_{3,3}
\end{bmatrix}
\begin{bmatrix}
    x_{1} \\\
    x_{2} \\\
    x_{3}
\end{bmatrix}

=

\begin{bmatrix}
    w_{1,1}x_{1} + w_{1,2}x_{2} + w_{1,3}x_{3} \\\
    w_{2,1}x_{1} + w_{2,2}x_{2} + w_{2,3}x_{3} \\\
    w_{3,1}x_{1} + w_{3,2}x_{2} + w_{3,3}x_{3}
\end{bmatrix}
```

Never mind the math. Matrices are an efficient data structure that also enable parallel computation (with Nvidia GPUs).

## 3. The Last Concept (Elephant in the Room) - Gradient Descend

If you consider an entire network as a single function $y = f(x)$, where x is an output, f is the network, and y as the output, the act of computing $y$ is called the 'Forward Pass'. For a particular set of parameters (or weights), you can calculate a loss by comparing it with the actual output you want. 

$$ Loss = f(x) - Y $$

$Y$ is the actual output you want, which is considered the target label in a labelled dataset.

The loss is a measure of how far off the network is from the actual output. The goal of training is to minimize the loss. 

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1062/1*KXhoClXogcckzwxfvIFReQ.png" alt="drawing" width="600"/>
</p>

You can use calculus to do this. Again, nevermind the math, the main thing to note is that you update the weight of each neuron by nudging it every so slightly from its current value to see which way you go will reduce the loss graph above. This is called the 'Backward Pass'. Nudging the values is called calculating the $Gradient$ for each neuron.

Intuitively, performing the backward pass tells you how much each neuron contributes to the total loss.

So, if I had to write this in Psuedo Code, it would look something like this:

```python
dataset = "a 2000 row dataset with x and target y values"
weights = "initialize a random set of values in matrices"

for data in datapoint:
    model_output = conduct_forward_pass(weights, data)
    loss = model_output - data.target
    weights = conduct_backward_pass(weights, loss)
```

As you can see, with the for loop, we visit each datapoint once. That's usually not enough. So if we wrap this in another for loop so that it visits the entire dataset $N$ times, we call this an $Epoch$. As you can imagine, humans too have to visit a piece of information multiple times to remember it.

```python
dataset = "a 2000 row dataset with x and target y values"
weights = "initialize a random set of values in matrices"

for epoch in range(0, N):
    for data in datapoint:
        model_output = conduct_forward_pass(weights, data)
        loss = model_output - data.target
        weights = conduct_backward_pass(weights, loss)
```

Done! Thats the main idea behind neural networks.

## 4. The Epilogue - Batched Training

So we trained a neural network where in each loop in the for loop (a single training cycle in other words) we visited a single datapoint. Just to bring up a main point with Neural Networks, we are trying to generalize them over a dataset so that they are good at predicting outputs for a wide variety of data that they haven't seen without overfitting them to the training data. One way is to batch your data and then train it.

How does batching help? Instead of updating the weights for each datapoint, you can take the average loss of a batch of datapoints and then update the weights once. A batch size of 4 means that you visit 4 datapoints, calculate the loss for each, average it, and then update the weights. That's a single for loop (training cycle).

In Code, we just add an extra dimension to the input data $X$. And because matrices allow for parallel computing, batching doesn't actually have any overhead other than having extra memory requirements for the extra dimension.

If the math is interesting, here is the simplified version of a batched forward pass:


Let's first simplify to having 2 'linear' layers where a neuron in one particular layer is connected to every neuron in the next layer.

```math
\begin{bmatrix}
    w_{1,1} & w_{1,2} \\\
    w_{2,1} & w_{2,2}
\end{bmatrix}

\begin{bmatrix}
    x_{1,1} & x_{1,2} & x_{1,3} & x_{1,4} \\\
    x_{2,1} & x_{2,2} & x_{2,3} & x_{2,4}
\end{bmatrix}

=

\begin{bmatrix}
    w_{1,1}x_{1,1} + w_{1,2}x_{2,1} & w_{1,1}x_{1,2} + w_{1,2}x_{2,2} & w_{1,1}x_{1,3} + w_{1,2}x_{2,3} & w_{1,1}x_{1,4} + w_{1,2}x_{2,4} \\\
    w_{2,1}x_{1,1} + w_{2,2}x_{2,1} & w_{2,1}x_{1,2} + w_{2,2}x_{2,2} & w_{2,1}x_{1,3} + w_{2,2}x_{2,3} & w_{2,1}x_{1,4} + w_{2,2}x_{2,4}
\end{bmatrix}
```

Here, each additional column in the input matrix is a datapoint.

### The End
