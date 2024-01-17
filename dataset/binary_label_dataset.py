import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class BinaryLabelDataset(Dataset):
    def __init__(self, seeg_file='../data/downsampled_seeg.npy', label_file='../data/seconds_with_Tony30.npy',
                 split='train', train_ratio=0.7, test_ratio=0.15):
        super(BinaryLabelDataset).__init__()
        self.split = split
        assert train_ratio + test_ratio <= 1, "The sum of train_ratio and test_ratio must be less than or equal to 1"

        # Load the data
        seeg_data = np.load(seeg_file).transpose(1, 0)
        label_data = np.load(label_file)

        # Truncate the unmatched data
        valid_second = min(int(seeg_data.shape[0] / 90), label_data.shape[0])
        seeg_data = seeg_data[:valid_second * 90, :].reshape(-1, 84, 90)  # Reshape to (valid_second, 84, 90)
        label_data = label_data[:valid_second]    # Reshape to (valid_second,)

        # Truncate the data where the label is -1
        seeg_data = seeg_data[label_data != -1, :, :]
        label_data = label_data[label_data != -1]

        # Compute the number of samples for train and test+val
        total_sample_num = seeg_data.shape[0]
        train_num = int(total_sample_num * train_ratio)
        test_val_num = total_sample_num - train_num

        # Stratified split for train and test+val
        seeg_train, seeg_test_val, label_train, label_test_val = train_test_split(
            seeg_data, label_data, train_size=train_num, random_state=42, stratify=label_data)

        # Further split test+val into test and val
        test_num = int(test_val_num * (test_ratio / (1 - train_ratio)))
        seeg_test, seeg_val, label_test, label_val = train_test_split(
            seeg_test_val, label_test_val, train_size=test_num, random_state=42, stratify=label_test_val)

        # Assign data based on split
        if self.split == 'train':
            self.seeg_data = seeg_train
            self.label_data = label_train
        elif self.split == 'test':
            self.seeg_data = seeg_test
            self.label_data = label_test
        elif self.split == 'val':
            self.seeg_data = seeg_val
            self.label_data = label_val

        self.total_num = len(self.seeg_data)

        print(f'Initialized {split} dataset with {self.total_num} samples')

    def __getitem__(self, index):
        seeg = torch.from_numpy(self.seeg_data[index, :, :]).float()
        label = torch.tensor([self.label_data[index]]).float()
        return seeg, label

    def __len__(self):
        return self.total_num


if __name__ == "__main__":
    seeg_file = '../data/downsampled_seeg.npy'
    label_file = '../data/seconds_with_Tony30.npy'

    dataset = BinaryLabelDataset(seeg_file=seeg_file, label_file=label_file, split='train')

    print("Checking the shape of the data...")
    for idx in range(len(dataset)):
        data = dataset[idx]
        assert data[0].shape == (84, 90), "The sEEG data must be of shape (84, 3)"
        assert data[1].shape == (1,), "The label must be of shape (1,)"
    print("The shape of the data is correct")

    labels = dataset.label_data
    num_pos = np.sum(labels)
    num_neg = len(labels) - num_pos
    print(f'Number of positive samples: {num_pos}')
    print(f'Number of negative samples: {num_neg}')

    dataset = BinaryLabelDataset(seeg_file=seeg_file, label_file=label_file, split='val')
    labels = dataset.label_data
    num_pos = np.sum(labels)
    num_neg = len(labels) - num_pos
    print(f'Number of positive samples: {num_pos}')
    print(f'Number of negative samples: {num_neg}')

    dataset = BinaryLabelDataset(seeg_file=seeg_file, label_file=label_file, split='test')
    labels = dataset.label_data
    num_pos = np.sum(labels)
    num_neg = len(labels) - num_pos
    print(f'Number of positive samples: {num_pos}')
    print(f'Number of negative samples: {num_neg}')

