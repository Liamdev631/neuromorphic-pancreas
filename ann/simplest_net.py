import torch.nn as nn

# Define a simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Reshape input tensor if needed
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        x = self.relu1(x)
        x = self.fc2(x)

        x = self.relu2(x)
        x = self.fc3(x)

        return x