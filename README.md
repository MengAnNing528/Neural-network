# Neural-network

How It Works - Step by Step Explanation
1. Data Preparation
Loading CSV: The dataset is loaded from a CSV file using pandas

Preprocessing:

Features are normalized using StandardScaler

Labels are converted to one-hot encoding (e.g., class 0 becomes [1,0,0], class 1 becomes [0,1,0])

Train/Test Split: Data is split into training and testing sets

2. Neural Network Architecture
text
Input Layer (4 nodes) → Hidden Layer (10 nodes) → Output Layer (3 nodes)
Input Layer: 4 nodes (for Iris dataset features)

Hidden Layer: 10 nodes with sigmoid activation

Output Layer: 3 nodes with softmax activation (for 3 classes)

3. Forward Propagation
python
def forward(self, X):
    self.z1 = np.dot(X, self.W1) + self.b1        # Linear transformation
    self.a1 = self.sigmoid(self.z1)               # Activation
    self.z2 = np.dot(self.a1, self.W2) + self.b2  # Linear transformation
    self.a2 = self.softmax(self.z2)               # Output probabilities
    return self.a2
Step 1: Multiply inputs by weights, add bias (z1 = X·W1 + b1)

Step 2: Apply sigmoid activation (a1 = σ(z1))

Step 3: Multiply hidden layer outputs by weights, add bias (z2 = a1·W2 + b2)

Step 4: Apply softmax to get probabilities (a2 = softmax(z2))

4. Backward Propagation (Learning)
python
def backward(self, X, y, output):
    # Calculate error at output
    dz2 = output - y
    
    # Update weights and biases for output layer
    dW2 = np.dot(self.a1.T, dz2)
    db2 = np.sum(dz2, axis=0)
    
    # Calculate error for hidden layer
    dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.a1)
    
    # Update weights and biases for hidden layer
    dW1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0)
    
    # Apply updates
    self.W2 -= learning_rate * dW2
    self.b2 -= learning_rate * db2
    self.W1 -= learning_rate * dW1
    self.b1 -= learning_rate * db1
Uses chain rule to calculate how much each weight contributed to the error

Adjusts weights to minimize the error

5. Training Loop
Repeatedly performs forward and backward propagation

Calculates loss (cross-entropy) to measure performance

Updates weights using gradient descent

Key Concepts Explained
Activation Functions
Sigmoid: Squashes values between 0-1, used in hidden layers

Softmax: Converts outputs to probabilities that sum to 1, used in output layer for classification

Loss Function
Cross-Entropy: Measures how well the predicted probabilities match the actual labels

Formula: -Σ(y_true * log(y_pred))

Gradient Descent
Adjusts weights in the direction that reduces error

Learning rate controls how big the steps are

Sample CSV Format
The code expects a CSV file where:

Each row represents one sample

All columns except the last are features

The last column contains the target labels

Example:

csv
sepal_length,sepal_width,petal_length,petal_width,target
5.1,3.5,1.4,0.2,0
4.9,3.0,1.4,0.2,0
7.0,3.2,4.7,1.4,1
6.3,3.3,6.0,2.5,2
This example provides a complete, working neural network that you can adapt for your own CSV datasets by modifying the input size, hidden layers, and output size according to your data.

