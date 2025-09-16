import torch
import torch.nn as nn
import torch.optim as optim

# Tiny dataset: XOR problem
"""
in here X is the input tensor and y is the target tensor.
X = [[0,0],[0,1],[1,0],[1,1]]

0 XOR 0 → 0
0 XOR 1 → 1
1 XOR 0 → 1
1 XOR 1 → 0

so y = [[0],[1],[1],[0]]
therefore the model should learn to predict y from X.

this model has 2 layers:
- first layer: 2 inputs to 4 hidden neurons with tanh activation
- second layer: 4 hidden neurons to 1 output with sigmoid activation

in these dense models (fully connected layers), each neuron in one layer is connected to every neuron in the next layer.
and every input is connected to every neuron in the first layer.

Parameters = (in_features + 1) x out_features
- in_features = number of inputs to the layer (weights)
- out_features = number of neurons in the layer
- +1 is for the bias term

For first layer:
- in_features = 2 (2 inputs)
- out_features = 4 (4 hidden neurons)
- Parameters = (2 + 1) x 4 = 12

For second layer:
- in_features = 4 (4 hidden neurons)
- out_features = 1 (1 output)
- Parameters = (4 + 1) x 1 = 5

- Total Parameters = 12 + 5 = 17
"""

X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

# Tiny model
class TinyNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)   # 2 inputs -> 4 hidden neurons
        self.fc2 = nn.Linear(4, 1)   # 4 hidden -> 1 output
    
    def forward(self, x):
        # first used ReLU but it didn't work well for XOR
        # changed to tanh and it worked better
        x = torch.tanh(self.fc1(x)) # Activation function
        x = torch.sigmoid(self.fc2(x))
        return x

model = TinyNN()
criterion = nn.BCELoss()            # Binary classification
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Test
with torch.no_grad():
    print("Predictions:", model(X).round())
