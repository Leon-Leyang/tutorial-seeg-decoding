import torch
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss


def eval_binary_label_model(model, criterion, data_loader, device, split):
    model.eval()

    total = 0
    correct = 0

    total_loss = 0

    with torch.no_grad():
        for seeg, label in data_loader:
            seeg, label = seeg.to(device), label.to(device)

            # Forward
            pred = model(seeg)

            # Compute loss
            loss = criterion(pred, label)
            total_loss += loss.item()

            # Convert predictions to binary
            pred_bin = (pred > 0.5).int()

            # Update total and correct
            total += label.shape[0]
            correct += (pred_bin == label).sum().item()

    # Compute average loss
    avg_loss = total_loss / len(data_loader)
    print(f'{split} average loss per batch: {avg_loss:.3f}')

    # Compute accuracy
    acc = correct / total
    print(f'{split} accuracy: {acc * 100:.2f}%')
    return acc, avg_loss


def eval_multi_label_model(model, data_loader, device, split):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for seeg, label in data_loader:
            seeg, label = seeg.to(device), label.to(device)

            # Forward
            pred = model(seeg)

            # Convert predictions to binary
            pred_bin = pred > 0.5

            all_labels.append(label.cpu())
            all_predictions.append(pred_bin.cpu())

    # Concatenate all batches
    all_labels = torch.cat(all_labels, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)

    # Calculate metrics
    f1 = f1_score(all_labels, all_predictions, average='macro')
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=1)
    recall = recall_score(all_labels, all_predictions, average='macro')
    hammingloss = hamming_loss(all_labels, all_predictions)

    print(f'{split} f1 score: {f1 * 100:.2f}%, precision: {precision * 100:.2f}%, recall: {recall * 100:.2f}%, '
          f'hamming loss: {hammingloss * 100:.2f}%')

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'hamming_loss': hammingloss
    }
