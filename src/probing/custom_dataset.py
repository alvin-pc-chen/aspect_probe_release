from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, labels, tensors, dim):
        self.hidden_state = tensors
        self.label = labels
        self.hs_dim = dim

    def __len__(self):
        return len(self.hidden_state)

    def __getitem__(self, idx):
        return self.label[idx], self.hidden_state[idx]