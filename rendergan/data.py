import h5py
import torch


class DatasetFromHdf5(torch.utils.data.Dataset):
    def __init__(self, file_path, data_label, target_label):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get(data_label)
        self.target = hf.get(target_label)

    def __getitem__(self, index):
        batch = torch.from_numpy(self.data[index,:,:, :]).float()
        return batch, batch
        
    def __len__(self):
        return self.data.shape[0]