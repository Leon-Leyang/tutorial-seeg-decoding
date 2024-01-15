import os
import numpy as np
import torch
from torch.utils.data import Dataset
from skmultilearn.model_selection import IterativeStratification


class MultiLabelDataset(Dataset):
    def __init__(self, seeg_file='../data/seeg.npy', label_folder='../data/presence_of_faces', split='train',
                 train_ratio=0.7, test_ratio=0.15,
                 chars_of_interest=['Amit', 'Dolores', 'Don', 'George', 'Kindell', 'Oleg', 'Tony']):
        super(MultiLabelDataset).__init__()
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
        assert len(set([label.shape[0] for label in
                        labels])) == 1, "The labels are not the same length"  # Check if the labels are the same length
        label_data = np.concatenate(labels, axis=-1)

        # The valid time is the minimum of time in seconds of the sEEG data and the label data
        valid_time = min(int(seeg_data.shape[0] / 1024), label_data.shape[0])

        # Truncate the data
        seeg_data = seeg_data[:valid_time * 1024, :].reshape(-1, 84, 1024)  # Reshape to (valid_time, 84, 1024)
        label_data = label_data[:valid_time, :]  # Reshape to (valid_time, number of characters)

        # Create stratified splits
        # Split train and test+val
        stratifier = IterativeStratification(n_splits=2, order=1,
                                             sample_distribution_per_fold=[1.0 - train_ratio, train_ratio])
        train_indexes, test_val_indexes = next(stratifier.split(seeg_data, label_data))

        seeg_train = seeg_data[train_indexes]
        label_train = label_data[train_indexes]
        seeg_test_val = seeg_data[test_val_indexes]
        label_test_val = label_data[test_val_indexes]

        # Split test and val
        test_ratio_adjusted = test_ratio / (1 - train_ratio)
        stratifier = IterativeStratification(n_splits=2, order=1,
                                             sample_distribution_per_fold=[1.0 - test_ratio_adjusted,
                                                                           test_ratio_adjusted])
        test_indexes, val_indexes = next(stratifier.split(seeg_test_val, label_test_val))

        seeg_test = seeg_test_val[test_indexes]
        label_test = label_test_val[test_indexes]
        seeg_val = seeg_test_val[val_indexes]
        label_val = label_test_val[val_indexes]

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
        label = torch.tensor(self.label_data[index]).float()
        return seeg, label

    def __len__(self):
        return self.total_num


if __name__ == "__main__":
    seeg_file = '../data/seeg.npy'
    label_folder = '../data/presence_of_faces'

    dataset = MultiLabelDataset(seeg_file=seeg_file, label_folder=label_folder, split='train')

    print("Checking the shape of the data...")
    for idx in range(len(dataset)):
        data = dataset[idx]
        assert data[0].shape == (84, 1024), "The sEEG data must be of shape (84, 1024)"
        assert data[1].shape == (7,), "The label must be of shape (7,)"
    print("The shape of the data is correct")

    dataset = MultiLabelDataset(split='val')

    dataset = MultiLabelDataset(split='test')
