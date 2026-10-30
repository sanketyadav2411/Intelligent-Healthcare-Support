import math
from data_preprocessing import load_and_preprocess


# Sigmoid activation function
def sigmoid(x):
    # Prevent overflow in exp()
    x = max(-500, min(500, x))
    return 1 / (1 + math.exp(-x))


# Load preprocessed dataset
data = load_and_preprocess()

X_train = data["nn"]["X_train"].values.tolist()
X_test = data["nn"]["X_test"].values.tolist()

y_train = data["y_train"]
y_test = data["y_test"]

# Convert target values into a flat list of integers
if hasattr(y_train, "values"):
    y_train = y_train.values.tolist()

if hasattr(y_test, "values"):
    y_test = y_test.values.tolist()

y_train = [
    int(value[0] if isinstance(value, list) else value)
    for value in y_train
]

y_test = [
    int(value[0] if isinstance(value, list) else value)
    for value in y_test
]


# Initialize weights and biases
n = len(X_train[0])
w1 = [0.01] * n
b1 = 0.0

w2 = 0.1
b2 = 0.0

learning_rate = 0.01
epochs = 1000
m = len(X_train)


# Training the neural network
for epoch in range(epochs):

    dw1 = [0.0] * n
    db1 = 0.0
    dw2 = 0.0
    db2 = 0.0

    for i in range(m):

        # Forward propagation: input to hidden layer
        z1 = 0.0

        for j in range(n):
            z1 += X_train[i][j] * w1[j]

        hidden = sigmoid(z1 + b1)

        # Hidden layer to output layer
        output = sigmoid(hidden * w2 + b2)

        # Calculate error
        error = output - y_train[i]

        # Backpropagation: output layer
        dw2 += error * hidden
        db2 += error

        # Backpropagation: hidden layer
        hidden_error = error * w2 * hidden * (1 - hidden)

        for j in range(n):
            dw1[j] += hidden_error * X_train[i][j]

        db1 += hidden_error

    # Calculate average gradients
    for j in range(n):
        dw1[j] /= m

    db1 /= m
    dw2 /= m
    db2 /= m

    # Update weights and biases
    for j in range(n):
        w1[j] -= learning_rate * dw1[j]

    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2

    if (epoch + 1) % 100 == 0:
        print("Epoch:", epoch + 1)


# Test the neural network
print("\nPredictions on test data:")

correct = 0

for i in range(len(X_test)):

    z1 = 0.0

    for j in range(n):
        z1 += X_test[i][j] * w1[j]

    hidden = sigmoid(z1 + b1)
    output = sigmoid(hidden * w2 + b2)

    predicted_class = 1 if output >= 0.5 else 0

    if predicted_class == y_test[i]:
        correct += 1

    if i < 10:
        print(
            "Actual:", y_test[i],
            "| output neuron value:", round(output, 3),
            "| Predicted:", predicted_class
        )


# Calculate accuracy
accuracy = (correct / len(X_test)) * 100
print("accuracy :", accuracy)


