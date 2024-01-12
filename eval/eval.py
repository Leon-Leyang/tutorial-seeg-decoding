import torch
from tqdm import tqdm


def eval(model, data_loader, device, split):
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for seeg, label in tqdm(data_loader):
            seeg = seeg.to(device)
            label = label.to(device)

            # Forward
            pred = model(seeg)

            # Convert predictions to binary
            pred_bin = (pred > 0.5).int()

            # Update total and correct
            total += label.shape[0]
            correct += (pred_bin == label).sum().item()

    # Compute accuracy
    acc = correct / total
    print(f'{split} Accuracy: {acc:.3f}')
    return acc
