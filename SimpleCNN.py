import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the model weights (example)
def load_model(model, checkpoint_path):
    # if GPU then load model with GPU else load model with CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model.load_state_dict(torch.load(checkpoint_path))
        print("Model is Currently using GPU")
    else:
        device = torch.device('cpu')
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Model is Currently using CPU")
    model.eval()
    return model

