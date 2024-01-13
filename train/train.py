import torch.nn.functional as F


def train(model, optimizer, data_loader, device):
    model.train()

    for seeg, label in data_loader:
        seeg = seeg.to(device)
        label = label.to(device)

        # Forward
        pred = model(seeg)

        # Compute loss
        loss = F.binary_cross_entropy(pred, label)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()
