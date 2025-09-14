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
