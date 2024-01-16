def train(model, optimizer, criterion, data_loader, device):
    model.train()

    total_loss = 0

    for seeg, label in data_loader:
        seeg = seeg.to(device)
        label = label.to(device)

        # Forward
        pred = model(seeg)

        # Compute loss
        loss = criterion(pred, label)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f'Train average loss per batch: {avg_loss:.3f}')
    return avg_loss
