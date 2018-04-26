from torch.utils import data
import h5py

class HDF5Dataset(data.Dataset):
    def __init__(self, hdf5_list_file):
        self.datasets = []
        self.labels = []
        self.total_count = 0
        with open(hdf5_list_file) as f:
            hdf5_list = f.readlines()
            hdf5_list = [x.strip() for x in hdf5_list] 
        
        for f in hdf5_list:
            h5_file = h5py.File(f, 'r')
            x = h5_file['data']
            y = h5_file['label']

            self.datasets.append(x)
            self.labels.append(y)
            self.total_count += len(x)

    def __len__(self):
        return self.total_count
    
    def __getitem__(self, index):
        '''
        Suppose each hdf5 file has 7000 samples
        '''
        dataset_index = int(index / 7000)
        in_dataset_index = int(index % 7000)
        return (self.datasets[dataset_index][in_dataset_index], self.labels[dataset_index][in_dataset_index])