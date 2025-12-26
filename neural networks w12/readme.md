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

**Layer of Neurons**

![layer of neurons](https://github.com/user-attachments/assets/6c9f2b11-e0c8-4599-822e-3664dd691b32)

Image shows how multiple neurons work together in a single layer.

Input: [1, 2, 3, 2.5]

Weights: 4 x 3 = 12 total weights
- Neuron 0 weights: [0.2, 0.8, -0.5, 1.0]
- Neuron 1 weights: [0.5, -0.91, 0.26, -0.5]
- Neuron 2 weights: [-0.26, -0.27, 0.17, 0.87]

Biases: [2, 3, 0.5]

Calculation:

Neuron 0: (1 × 0.2) + (2 × 0.8) + (3 × -0.5) + (2.5 × 1.0) + 2 = 4.8. After activation → output0

Neuron 1: (1 × 0.5) + (2 × -0.91) + (3 × 0.26) + (2.5 × -0.5) + 3 = 1.21. After activation → output1

Neuron 2: (1 × -0.26) + (2 × -0.27) + (3 × 0.17) + (2.5 × 0.87) + 0.5 = 2.385. After activation → output2

Final Output: [output0, output1, output2]

**By Dot Product**

![dot product](https://github.com/user-attachments/assets/d5abc1bd-268d-4d49-b425-f45b56dae879)

```python
inputs = np.array(inputs)
weights = np.array([weights0, weights1, weights2])
biases = np.array([bias0, bias1, bias2])
outputs = np.dot(weights, input) + biases
```

**Difference between tensors and lists**

A list is a basic Python data structure that can hold any type of data in a sequence.

```python
list = [1, 2, 3, 4, 2]

mixed_list = [1, 'hello', 3.14, True]

matrix = [ [1, 2, 3]
           [4, 4, 6]
           [4, 8, 9] ]

cube = [ [ [1, 2], [3, 4] ],
         [  [5, 6], [7, 9] ] ]
```

Disadvantage: slow for mathematical operations.

A tensor is a multi-dimensional array optimized for numerical computations. 

```python
# 0D Tensor (Scaler) - A single number
scaler = np.array(5)
# shape: ()

# 1D Tensor (Vector) - A list of numbers
vector = np.array([2, 3, 4, 1, 5])
# shape: (5, )

# 2D Tensor (Matrix) - A table of numbers
matrix = np.array( [ [1, 2, 3],
                     [4, 5, 2],
                     [3, 9, 8]])
# shape: (3, 3)

# 3D Tensor - A cube of numbers
cube = np.array([[[1, 2], [3, 4]], 
                 [[5, 6], [7, 8]]])
# shape: (2, 2, 2) like (height, width, rgb channels)

# 4D Tensor - Multiple 3D tensors
batch = np.array( [[[[1, 2], [3, 4]], [[5, 3], [7, 4]]],
                   [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])
# shape: (2, 2, 2, 2) like a batch of color images
```
Key differences

List:
- Type: python built-in
- Speed: slow
- math operations: manual loop needed
- memory: more memory
- data type: mixed types allowed

Tensor:
- type: numpy, pytorch, tensorflow object
- speed: very fast (GPU accelerated)
- math operations: built-in (+ - x /)
- memory: optimized memory
- data type: single type (all int, all float)

Example:
```python
# using lists
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

outputs = []
for neuron_weights in weights:
   output = 0
   for i in range(len(inputs):
      output += inputs[i] * neuron_weights[i]
   outputs.append(output)

# using tensor
import numpy as np
inputs = np.array([1.0, 2.0, 3.0, 2.5])
weights = np.array([[0.2, 0.8, -0.5, 1.0],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]])

outputs = np.dot(weights, inputs) # only one line
```
Tensor shape in neural networks

input tensor: (batch_size, features)
- (32, 784) = 32 image, each with 784 pixels

weight tensor: (output_neurons, input_features)
- (10, 784) = 10 outputs, 784 inputs each

output tensor: (batch_size, output_neurons)
- (32, 10) = 32 predictions, 10 classes each

**How matrix multiplication works**

Rule: To multiply matrices, the number of columns in the first matrix must equal the number of rows in the second matrix.

Matrix A (m x n) x Matrix B (n x p) = Result (m x p)

![matrix multiplication](https://github.com/user-attachments/assets/5f749370-9e32-4817-bb63-e75188dfbda0)

Calculation Details

Step 1: Identify the row and column
- Row 2 from Matrix A: [0.9, -0.4, 0.7, 1.1]
- Column 2 from Matrix B: [1.5, 1.8, 0.7, 1.3]

Step 2: Multiply corresponding elements
- 0.9 * 1.5 = 1.35
- (-0.4) * 1.8 = -0.72
- 0.7 * 0.7 = 0.49
- 1.1 * 1.3 = 1.43

Step 3: Sum all products
- 1.35 + (-0.72) + 0.49 + 1.43 = 2.55

**Activation Function**

1. Linear Activation Function
   - The simplest activation function where the output equals the input.
   - Formauls: y = x

![linear fuction](https://github.com/user-attachments/assets/ccab452d-3c6d-477c-abb7-c3b89f8f0b86)

2. Sigmoid Activation Function
   - Squashes input values between 0 and 1.
   - Formaula: y = 1 / (1 + e^(-x))

![sigmoid](https://github.com/user-attachments/assets/f6789426-e5db-4d68-a19e-49de09a59590)

3. Rectified Linear Unit Activation Funtion
   - Returns 0 for negative inputs, and the inputs itself for positive inputs.
   - Formula: y = max(0, x)
  
![relu](https://github.com/user-attachments/assets/6d59b03e-e408-47cf-8e9b-4f1ed362b36a)

4. Step Activation Function
   - The original perceptron activation function. Outputs 0 or 1 based on threshold.
   - Formula: y = 1 if x > 0 else 0

![step function](https://github.com/user-attachments/assets/34758df4-114f-49fc-b398-ee546f06194b)

5. Softmax Function
   - Converts a vector of raw scores (logits) into a probability distribution. All output sum to 1.
   - Formaula: Softmax (xᵢ) = eˣⁱ / Σⱼ eˣʲ
  
![softmax](https://github.com/user-attachments/assets/2c3d5a0a-d5e0-47e7-88c2-8a360408807f)

**Accuracy Calculation**

Accuracy measures how often the model's predictions match the true labels. It is calculated as the fraction of correct predictions.

```python
predictions = np.argmax(softmax_output, axis=1) # take the class with highest probability
accuracy = np.mean(predictions == class_targets)
```

argmax() is a function that returns the indices of the maximum element along a specified axis in a given array.
