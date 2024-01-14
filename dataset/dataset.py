import os
import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, seeg_file='../data/seeg.npy', label_folder='../data/presence_of_faces', split='train',
                 train_ratio=0.7, test_ratio=0.15,
                 chars_of_interest=['Amit', 'Dolores', 'Don', 'George', 'Kindell', 'Oleg', 'Tony']):
        super(CustomDataset).__init__()
        self.split = split
        assert train_ratio + test_ratio <= 1, "The sum of train_ratio and test_ratio must be less than or equal to 1"

        # Load the data seeg data
        seeg_data = np.load(seeg_file).transpose(1, 0)

        # Load the labels and concatenate them
        labels = []
        for char in chars_of_interest:
            label_file = os.path.join(label_folder, f'seconds_with_{char}0.npy')
            label = np.load(label_file)
            labels.append(np.expand_dims(label, axis=-1))
        assert len(set([label.shape[0] for label in labels])) == 1, "The labels are not the same length"    # Check if the labels are the same length
        label_data = np.concatenate(labels, axis=-1)

        # The valid time is the minimum of time in seconds of the sEEG data and the label data
        valid_time = min(int(seeg_data.shape[0] / 1024), label_data.shape[0])

        # Truncate the data
        seeg_data = seeg_data[:valid_time * 1024, :].reshape(-1, 84, 1024)  # Reshape to (valid_time, 84, 1024)
        label_data = label_data[:valid_time, :]    # Reshape to (valid_time, number of characters)

        # Compute the number of samples for each split
        train_num = int(valid_time * train_ratio)
        test_num = int(valid_time * test_ratio)
        val_num = valid_time - train_num - test_num

        # Split the data
        if self.split == 'train':
            self.total_num = train_num
            self.seeg_data = seeg_data[:train_num, :, :]
            self.label_data = label_data[:train_num, :]
        elif self.split == 'test':
            self.total_num = test_num
            self.seeg_data = seeg_data[train_num:train_num + test_num, :, :]
            self.label_data = label_data[train_num:train_num + test_num, :]
        elif self.split == 'val':
            self.total_num = val_num
            self.seeg_data = seeg_data[train_num + test_num:, :, :]
            self.label_data = label_data[train_num + test_num:, :]

        print(f'Initialized {split} dataset with {self.total_num} samples')

    def __getitem__(self, index):
        seeg = torch.from_numpy(self.seeg_data[index, :, :]).float()
        label = torch.tensor(self.label_data[index]).float()
        return seeg, label

    def __len__(self):
        return self.total_num


if __name__ == "__main__":
    seeg_file = '../data/seeg.npy'
    label_folder = '../data/presence_of_faces'

    dataset = CustomDataset(seeg_file=seeg_file, label_folder=label_folder, split='train')
    print(f'The training dataset has {len(dataset)} samples')

    print("Checking the shape of the data...")
    for idx in range(len(dataset)):
        data = dataset[idx]
        assert data[0].shape == (84, 1024), "The sEEG data must be of shape (84, 1024)"
        assert data[1].shape == (7,), "The label must be of shape (7,)"
    print("The shape of the data is correct")

    dataset = CustomDataset(split='val')
    print(f'The validation dataset has {len(dataset)} samples')

    dataset = CustomDataset(split='test')
    print(f'The test dataset has {len(dataset)} samples')
