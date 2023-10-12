import pickle
from torch.utils.data import IterableDataset

# Test files generator
class NumberIterableDataset(IterableDataset):
    def __init__(self, n):
        self.n = n

    def __iter__(self):
        return iter(range(self.n))

class StringIterableDataset(IterableDataset):
    def __init__(self, n):
        self.n = n

    def __iter__(self):
        for i in range(self.n):
            yield f"string_{i}"

def save_dataset(filename: str, dataset:IterableDataset):
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":
    save_dataset('num-iterdataset-100.pkl', NumberIterableDataset(100))
    save_dataset('str-iterdataset-10.pkl', StringIterableDataset(10))
