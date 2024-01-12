import torch.nn as nn


class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(84 * 1024, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 84 * 1024)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


if __name__ == "__main__":
    import torch

    model = FCNN()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy data
    batch_size = 32
    seegs = torch.rand(batch_size, 84, 1024)
    preds = model(seegs)
    assert preds.shape == (batch_size, 1), "The output shape must be (batch_size, 1)"
