from torch.utils import data
import h5py

class HDF5Dataset(data.Dataset):

    def __init__(self, hdf5_list):
        self.datasets = []
        self.total_count = 0
        for f in hdf5_list:
           h5_file = h5py.File(f, 'r')
           dataset = h5_file['YOUR DATASET NAME']
           self.datasets.append(dataset)
           self.total_count += len(dataset)

    def __getitem__(self, index):
        '''
        Suppose each hdf5 file has 10000 samples
        '''
        dataset_index = index % 10000
        in_dataset_index = int(index / 10000)
        return self.datasets[dataset_index][in_dataset_index]

    def __len__(self):
        return len(self.total_count)