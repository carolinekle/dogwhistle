import torch

class DogwhistleDataset(torch.utils.data.Dataset):
    
    def __init__(self, encodings, labels):
        # store the tokenized inputs and integer labels
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        # how many rows total
        return len(self.labels)

    def __getitem__(self, idx):
        # get one row by index — returns a dict of tensors
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item