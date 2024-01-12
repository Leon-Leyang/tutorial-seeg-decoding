import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, split='train', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        super(CustomDataset).__init__()
        self.split = split
        assert train_ratio + val_ratio + test_ratio == 1, "The sum of the ratios must be 1"

        # Dummy data for now
        total_num = 10000

        if self.split == 'train':
            self.total_num = int(total_num * train_ratio)
        elif self.split == 'val':
            self.total_num = int(total_num * val_ratio)
        elif self.split == 'test':
            self.total_num = int(total_num * test_ratio)

    def __getitem__(self, index):
        # Dummy data for now
        seeg = torch.rand(84, 1024)
        label = torch.randint(0, 2, (1,)).float()
        return seeg, label

    def __len__(self):
        return self.total_num


if __name__ == "__main__":
    dataset = CustomDataset()
    print(f'The dataset has {len(dataset)} samples')

    print("Checking the shape of the data...")
    for idx in range(len(dataset)):
        data = dataset[idx]
        assert data[0].shape == (84, 1024), "The sEEG data must be of shape (84, 1024)"
        assert data[1].shape == (1,), "The label must be of shape (1,)"
    print("The shape of the data is correct")
