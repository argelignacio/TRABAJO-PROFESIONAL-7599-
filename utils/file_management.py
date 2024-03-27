import os
import numpy as np
import pickle as pkl
import pandas as pd

class FileManagement:
    def __init__(self, folder_path):
        self.folder_path = folder_path

        # Create folder if not exists
        os.makedirs(self.folder_path, exist_ok=True)


    def get_folder_path(self):
        return self.folder_path
    
    def join_path(self, file_name):
        return os.path.join(self.folder_path, file_name)
    
    def save_df(self, file_name, df):
        df.to_csv(os.path.join(self.folder_path, file_name), index=False)

    def save_npy(self, file_name, np_array):
        np.array(np_array).dump(os.path.join(self.folder_path, file_name))

    def save_pkl(self, file_name, obj):
        with open(os.path.join(self.folder_path, file_name), 'wb') as f:
            pkl.dump(obj, f)
    
    def load_pkl(self, file_name):
        with open(os.path.join(self.folder_path, file_name), 'rb') as f:
            return pkl.load(f)
        
    def load_npy(self, file_name):
        return np.load(os.path.join(self.folder_path, file_name), allow_pickle=True)
    
    def load_df(self, file_name):
        return pd.read_csv(os.path.join(self.folder_path, file_name))