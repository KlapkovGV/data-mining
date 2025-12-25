## Neural Networks

Neural Networks are a fundamental concept in machine learning. The distinction between a standard neural network and a deep neural networj lies in the architecture: deep networks conyain two or more hidden layers. These hidden layers perform the intermediate computations that transform input data into meaningful outputs. 

Mathematical model of the artificial neuron. Multiple inputs (x1, x2, ..., xn) represent incoming signals, each typically multiplied by a corresponding weight (w1, w2, ..., wn). A summation function aggregates these weighted inputs and adds a bias term (b), which allows the neuron to shift its activation threshold. An activation function then determines the output, and a single output passes to the next layer.

Neural Network Architecture

![Neural Network Architecture](https://github.com/user-attachments/assets/dc9f8fea-2302-4fe0-9722-d8022cc206aa)

This is a deep neural network with 5 layers (1 input, 3 hidden, 1 output). The layer size are: 10, 16, 16, 16, 2.

1. Input Layer
   - Takes 10 input features (x1, x2, ... x10)
   - No bias, no weights
   - 0 parameters
2. Hidden Layer 1
   - Receives 10 inputs from the previuos layer
   - Has 16 biases (one per neuron)
   - Has 10 x 16 = 160 weights (each of 10 inputs connects to each of 16 neurons)
   - Total: 160 + 16 = 176 parameters
3. Hidden Layer 2
   - Receives 16 inputs from hidden layer 1
   - Has 16 biases
   - Has 16 x 16 = 256 weights
   - Total: 256 + 16 = 272 parameters
4. Hidden Layer 3
   - Receives 16 inputs from hidden layer 2
   - Has 16 biases
   - Has 16×16 = 256 weights
   - Total: 256 + 16 = 272 parameters
5. Output layer
   - Receives 16 input from hidden layer 3
   - Has 2 biases
   - Has 16 x 2 = 32 weights
   - Total: 32 + 2 = 34 parameters
  
Calculation
- Weights: 160 + 256 + 256 + 32 = 704 weights
- Biases: 16 + 16 + 16 + 2 = 50 biases
- Total parameters: 704 + 50 = 754 parameters

**Artificial neuron**

![Artificial Neuron](https://github.com/user-attachments/assets/5d6f2513-80b0-4084-a793-6523491a350b)

Image shows how one artificial neuron processes information.

Computation with values

Inputs: [5.0, -2.0, 1.5]
Weights: [0.3, 0.6, -0.2]
Bias: 1.5

Formula: output = activation( Σ (input * weight) + bias)

Step 1: Multiply inputs by weights
- 5.0 * 0.3 = 1.5
- (-2.0) * 0.6 = -1.2
- 1.5 * (-0.2) = -0.3

Step 2: Sum all weighted inputs
- sum = 1.5 + (-1.2) + (-0.3) = 0

Step 3: Add bias
- 0.0 + 1.5 = 1.5

Step 4: Apply activation function (Sigmoid)
- Sigmoid(1.5) = 0.8176

Points:
- Weights determine how much each input influences the output. Positive weights increase the output, negative decrease it.
- Bias shifts the activation threshold. It allows the neuron to activate even when all inputs are zero.
- Activation function determines the final output format (cound be sigmoid, ReLU, step function, etc).

**Loss Curve Visualization**

![loss curve visualization](https://github.com/user-attachments/assets/d6d1c12f-4105-4a9e-aee8-3f927ff50bc8)

This curve shows the training and validation loss over time (epochs) during neural network training. Concepts:

1. Training Loss (red line): Measures how well the model performs on the training data. It consistently decreases, meaning the model is learning.
2. Validation Loss (blue line): Measures how well the model performs on data it has not seen during the training. This shows the model's ability to gemeralize.
3. The purple line (sweet spot): This marks the optimal point where validation loss is at minimum. This is where we should stop straining.
4. After the purple line (yellow zone):  Training loss continues to decrease (model gets better at training data). Validation loss starts to increase (model gets worse at new data). This is called **OVERFITTING** - the model memorizes training instead of learning general patterns.

Dataset splits:
- Train: Data used to train the model;
- Validation: Data used to tune the model and decide when to stop training;
- Test: Data used only once at the end to evaluate final performance.

Features:
- Overfitting Strength - how much the model overfits;
- Leaning Rate - how fast the model learn
- Current Epoch slider
