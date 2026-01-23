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

# Why non-linearity changes everything? A nonlinear (ReLU, sigmoid, tanh) adds a "bend" in the rule. 
- The network is no longer forced to use only straigh-line boundaries
- it can combine many bends to make curved / complex shapes.

<img width="216" height="234" alt="image" src="https://github.com/user-attachments/assets/e08f024d-90cd-4fde-af75-966c18a1dda8" />



