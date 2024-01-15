import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from models.multiple_label_fcnn import MultiLabelFCNN
from dataset.multi_label_dataset import MultiLabelDataset
from torch.utils.data import DataLoader
from train.train import train
from eval.eval import eval
from utils.model import set_seeds
from utils.data import get_chars_freq


def main(args):
    # Create datasets
    print('Initializing datasets...')
    seeg_file = '../data/seeg.npy'
    label_folder = '../data/presence_of_faces'
    train_ratio = 0.7
    test_ratio = 0.15
    train_dataset = MultiLabelDataset(seeg_file=seeg_file, label_folder=label_folder, split='train', train_ratio=train_ratio,
                                      test_ratio=test_ratio)
    val_dataset = MultiLabelDataset(seeg_file=seeg_file, label_folder=label_folder, split='val', train_ratio=train_ratio,
                                    test_ratio=test_ratio)
    test_dataset = MultiLabelDataset(seeg_file=seeg_file, label_folder=label_folder, split='test', train_ratio=train_ratio,
                                     test_ratio=test_ratio)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    chars_freq = get_chars_freq(train_dataset)

    bias_init = [-np.log(freq / (1 - freq)) for freq in chars_freq]
    model = MultiLabelFCNN(bias_init=bias_init)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Specify loss function
    inverse_chars_freq = [1 / freq for freq in chars_freq]
    criterion = torch.nn.BCELoss(weight=torch.tensor(inverse_chars_freq).to(device))

    # Create checkpoint directory
    os.makedirs('../ckpt', exist_ok=True)

    best_val_acc = 0

    # Train
    print('\nTraining...')
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch}')

        # Train for one epoch
        train(model, optimizer, criterion, train_loader, device)

    #     # Evaluate on validation set
    #     val_acc = eval(model, val_loader, device, 'Val')
    #     if val_acc > best_val_acc:
    #         best_val_acc = val_acc
    #         torch.save(model.state_dict(), f'../ckpt/best_{model.__class__.__name__}.pth')
    #         print(f'Saved best model in epoch {epoch}')
    #
    # # Evaluate on test set
    # print('\nTesting...')
    # model.load_state_dict(torch.load(f'../ckpt/best_{model.__class__.__name__}.pth'))
    # eval(model, test_loader, device, 'Test')


def get_args():
    parser = argparse.ArgumentParser(description='Train a model on sEEG data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    set_seeds(42)
    main(args)
