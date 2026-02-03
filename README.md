# Deep-Learning

Deep learning is a way to make computers learn patterns from data using lots of stacked layers of simple math units. 

Just like a simple neuron from the brain, deep learning uses neurons (lots of them) to do mathematical calculations. 

<img width="482" height="630" alt="image" src="https://github.com/user-attachments/assets/8745d695-bd0b-4083-89ed-743179f3d6cf" />

In biology, we know that dendrites receive the signal and carry it to the nucleus where the signal is processed and the action is taken. In computer science, scientists took the inspiration from this created a perceptron (single-layer neural network). 

<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/50e6a366-8a67-4fb4-8ef6-f22bea09de75" />

In biology, signal transfer occurs at a specialized junction called as synapse, through a process called neurotransmission. In computer science, once the signal is passed from input layer and an actiona is taken, the signal (output for that layer) goes into an output layer and is then used as an input for the next neuron. This way a deep neural network (multiple layers of perceptrons) is created. To avoid showing each perceptron every time, people use the name hidden layer, to justify that a perceptron is present there, which carries out function. 

<img width="1024" height="576" alt="image" src="https://github.com/user-attachments/assets/673cfa79-338c-41b6-8403-2744a0286e14" />


A neuron (in a neural network) is a small function that:
1) takes numbers in (features)
2) computes a weighted sum + a bias
3) passes it through a non-linear function (activation)

Mathematically: 𝑦 = 𝜎(𝑤1𝑥1+𝑤2𝑥2+⋯+𝑤𝑛𝑥𝑛+𝑏)

Where, 
Weights (w) = what the neuron “cares about”
Bias (b) = shifts the decision threshold
Activation (σ) = makes it non-linear (ReLU, sigmoid, etc.)

# Why the activation matters? 
- Without nonlinearity, stacking layers collapses into one big linear function, and you can’t model complex patterns.

# What is a linear function? 
- A linear layer is basically input function -> multiple by some number -> add them up
- For example 2xx + 3xy + 1 (This will always make straight line boundaries between classes).

# What happens when you stack only linear layers? 
- Let's say we have two linear layers:
1) First: transform input -> some intermediate number
2) Second: transform the intermediate number -> output

- But this can be done with one big step only, we don't need two steps. Thus, stacking layers don't add real expressive power.

# Why non-linearity changes everything? 
- A nonlinear (ReLU, sigmoid, tanh) adds a "bend" in the rule. 
- The network is no longer forced to use only straigh-line boundaries
- it can combine many bends to make curved / complex shapes.

<img width="216" height="234" alt="image" src="https://github.com/user-attachments/assets/e08f024d-90cd-4fde-af75-966c18a1dda8" />

# How does network learn? 
- A neural network learns by adjusting its weights (parameters) so its predictions get closer to the correct answer. Think about it like turning thousands of knobs to get the correct answer.

# How do we make the loss small? 
- Firstly, loss is like a penalty score, given to each neuron. This helps it to understand how off it is compared to the actual result. Just like Mean Squared Error in regression or Cross Entropy loss during classification. 
- To reduce the loss, we have to find weights that minimize average loss over the training data.

<img width="1995" height="1331" alt="image" src="https://github.com/user-attachments/assets/e19d2554-1635-470a-85d5-dcba8894b68a" />

- The slope of the graph (x, y axis: different weight settings, height: loss value) will tell use which way is downhill and help us understand the global minima. 
- The direction of downhill is given by the gradient.
- Gradient: A verctor slope; if you increase this weight by a tiny bit, does loss go up or down, and by how much is given by gradient.

 Then you update weights like:

𝑤 ← 𝑤 − 𝜂 ⋅ ∂𝐿/∂𝑤

∂w/∂L = slope of loss w.r.t that weight; η (learning rate) = step size

- So: to make loss smaller, step a little opposite the slope.

# What is gradient descent? 
- Gradient descent is an optimisation algorithm used to minimize the model's error by adjusting its parameters. It helps the model to learn the best possible weight for better prediction. 

# How is gradient and gradient descent different? 
- The gradient is a vector of partial derivatives showing a function's steepest ascent, like the slope of a hill; Gradient Descent is an iterative optimization algorithm that repeatedly takes steps in the opposite (negative) direction of the gradient to find the minimum (lowest point) of a cost function.

# Types of gradients: 
- It depends on how the gradient is computed / estimated during training process.
1) True gradient descent / Batch Gradient Descent:
- Compute the gradient of the loss using all training examples.
- Less noisy, more accurate; expensive.

2) Stochastic gradient descent (SGD):
- Computes gradient using one example.
- Noisy estimates of true gradient; cheap and can help in exploration.

3) Mini-batch gradient:
- Compute graadient on a small batch of examples.
- cheaper than true gradient; but noisy than SGD

- Mini-batch gradient is an estimate of the true gradient.

# The learning loop
- Forward pass: Feed an input through the network -> get a prediction.
- Compute loss: Compare prediction vs trugh -> Produce a single number that says "how bad".
- Backpropagation: Figure out how much each weight contributed to the error.
- Update weights (gradient descent): Nudge each weight in the direction that reduces the loss.

- This step repeats over many iterations -> loss goes down -> model improves.

# Problem solvong
- Now that we know how the network works, let's see it in action. We have input x (numbers) and the output we want y (a class, number, sequence). A neural network learn a function f(x) ≈ y by tuning weights using backprop.

- **Artificial Neural Network (ANN)**:
- Ann is basically a fully-connected (dense) neural network, also called an MLP (Multi-Layer Perceptron).
- Input layer -> Hiiden layer(s) -> Output layer: Each neuron connects to all neurons in the next layer.

- Pros:
- Tabular data; simple signals when the structure is not spatial.

- Limitations:
- If images are given as input, the nearby pixels are lost
- A lot of parameters (overfitting risk)
- It doesn't scale well to high-res images.

# Other types of NN: 
1) Feedforward Network
- No loops, information flows one way. Statis input -> Output

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/0bb9dfb0-e4f6-422c-93f6-1084ea89a2f6" />

2) Recurrent Neural Network (RNN):
- Built for sequences (time series, text)
- Has feedback connection (memory)
- Processes input step-by-step over time

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/6795ee31-075f-46d1-9d09-e6d3646c4b12" />

The fundamental processing unit in RNN is a Recurrent Unit. They hold a hidden state that maintains information about previous inputs in a sequence. Recurrent units can "remember" information from prior steps by feeding back their hidden state, allowing them to capture dependencies across time.

In RNNs the hidden state Hi is calculated for every input Xi to retain sequential dependencies. The computations follow these core formulas:

1. Hidden State Calculation:

h = σ(U⋅X+W⋅ht−1+B)

Here:
h represents the current hidden state.
U and W are weight matrices.
B is the bias.

2. Output Calculation:
Y=O(V⋅h+C)

The output 
Y is calculated by applying O an activation function to the weighted hidden state where 
V and C represent weights and bias.

3. Overall Function:

Y=f(X,h,W,U,V,B,C)
 
This function defines the entire RNN operation where the state matrix 
S holds each element si representing the network's state at each time step i.

How does RNN work?
At each time step RNNs process units with a fixed activation function. These units have an internal hidden state that acts as memory that retains information from previous time steps. This memory allows the network to store past knowledge and adapt based on new inputs.

3) Autoencoders:
- Learn to compress and reconstruct data:
- (encoder) Input -> compact presentation -> Reconstruction (decoder)

<img width="647" height="400" alt="image" src="https://github.com/user-attachments/assets/b8c53cb9-441b-425d-876a-c6b5fdb7c472" />

Use case: Denoising, anomaly detection, representation learning. 

4) Graph Neural Network (GNN):
- Built for data that is node and edges (graph)
- Use cases: Molecules; social network; knowledge graphs

5) Transformers:
- Originally used for language recognition, not used for LLMs
- Idea: Attention lets every token relate to every other token.
- Use case: LLMs, Multimodal models.

<img width="824" height="690" alt="image" src="https://github.com/user-attachments/assets/67a1de9c-2f5a-4520-bb72-7e051ef77ae3" />

