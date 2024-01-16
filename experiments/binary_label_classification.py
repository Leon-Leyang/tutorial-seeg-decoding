import os
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from models.binary_label_fcnn import BinaryLabelFCNN
from dataset.binary_label_dataset import BinaryLabelDataset
from torch.utils.data import DataLoader
from train.train import train
from eval.eval import eval_binary_label_model
from utils.model import set_seeds


def main(args):
    # Create datasets
    print('Initializing datasets...')
    seeg_file = '../data/downsampled_seeg.npy'
    label_file = '../data/new_presence_of_faces/frames_with_Tony.npy'
    train_ratio = 0.7
    test_ratio = 0.15
    train_dataset = BinaryLabelDataset(seeg_file=seeg_file, label_file=label_file, split='train',
                                       train_ratio=train_ratio, test_ratio=test_ratio)
    val_dataset = BinaryLabelDataset(seeg_file=seeg_file, label_file=label_file, split='val', train_ratio=train_ratio,
                                     test_ratio=test_ratio)
    test_dataset = BinaryLabelDataset(seeg_file=seeg_file, label_file=label_file, split='test', train_ratio=train_ratio,
                                      test_ratio=test_ratio)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    model = BinaryLabelFCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Specify loss function
    criterion = F.binary_cross_entropy

    # Create checkpoint directory
    os.makedirs('../ckpt', exist_ok=True)

    best_val_acc = 0

    # Train
    print('\nTraining...')
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch}')

        # Train for one epoch
        train(model, optimizer, criterion, train_loader, device)

        # Evaluate on validation set
        val_acc = eval_binary_label_model(model, val_loader, device, 'Val')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'../ckpt/best_{model.__class__.__name__}.pth')
            print(f'Saved best model in epoch {epoch}')

    # Evaluate on test set
    print('\nTesting...')
    model.load_state_dict(torch.load(f'../ckpt/best_{model.__class__.__name__}.pth'))
    eval_binary_label_model(model, test_loader, device, 'Test')


def get_args():
    parser = argparse.ArgumentParser(description='Train a model on sEEG data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    set_seeds(42)
    main(args)
