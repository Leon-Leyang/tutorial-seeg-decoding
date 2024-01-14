import numpy as np
import torch
from torch.utils.data import Dataset


class BinaryLabelDataset(Dataset):
    def __init__(self, seeg_file='../data/seeg.npy', label_file='../data/presence_of_faces/seconds_with_Tony0.npy',
                 split='train', train_ratio=0.7, test_ratio=0.15):
        super(BinaryLabelDataset).__init__()
        self.split = split
        assert train_ratio + test_ratio <= 1, "The sum of train_ratio and test_ratio must be less than or equal to 1"

        # Load the data
        seeg_data = np.load(seeg_file).transpose(1, 0)
        label_data = np.load(label_file)

        # The valid time is the minimum of time in seconds of the sEEG data and the label data
        valid_time = min(int(seeg_data.shape[0] / 1024), label_data.shape[0])

        # Truncate the data
        seeg_data = seeg_data[:valid_time * 1024, :].reshape(-1, 84, 1024)  # Reshape to (valid_time, 84, 1024)
        label_data = label_data[:valid_time]    # Reshape to (valid_time,)

        # Compute the number of samples for each split
        train_num = int(valid_time * train_ratio)
        test_num = int(valid_time * test_ratio)
        val_num = valid_time - train_num - test_num

        # Split the data
        if self.split == 'train':
            self.total_num = train_num
            self.seeg_data = seeg_data[:train_num, :, :]
            self.label_data = label_data[:train_num]
        elif self.split == 'test':
            self.total_num = test_num
            self.seeg_data = seeg_data[train_num:train_num + test_num, :, :]
            self.label_data = label_data[train_num:train_num + test_num]
        elif self.split == 'val':
            self.total_num = val_num
            self.seeg_data = seeg_data[train_num + test_num:, :, :]
            self.label_data = label_data[train_num + test_num:]

        print(f'Initialized {split} dataset with {self.total_num} samples')

    def __getitem__(self, index):
        seeg = torch.from_numpy(self.seeg_data[index, :, :]).float()
        label = torch.tensor([self.label_data[index]]).float()
        return seeg, label

    def __len__(self):
        return self.total_num


if __name__ == "__main__":
    seeg_file = '../data/seeg.npy'
    label_file = '../data/presence_of_faces/seconds_with_Tony0.npy'

    dataset = BinaryLabelDataset(seeg_file=seeg_file, label_file=label_file, split='train')
    print(f'The training dataset has {len(dataset)} samples')

    print("Checking the shape of the data...")
    for idx in range(len(dataset)):
        data = dataset[idx]
        assert data[0].shape == (84, 1024), "The sEEG data must be of shape (84, 1024)"
        assert data[1].shape == (1,), "The label must be of shape (1,)"
    print("The shape of the data is correct")

    dataset = BinaryLabelDataset(split='val')
    print(f'The validation dataset has {len(dataset)} samples')

    dataset = BinaryLabelDataset(split='test')
    print(f'The test dataset has {len(dataset)} samples')
