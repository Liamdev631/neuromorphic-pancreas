from zipfile import ZipFile
from torch.utils.data import Dataset
import pandas as pd
import torch

class GlucoseDataset(Dataset):
    def __init__(self, zip_file_path, sub_directory, sample_length=16, transform=None):
        self.zip_file = ZipFile(zip_file_path, 'r')
        self.file_list = self.zip_file.namelist()  # Get list of files in the zip
        # Remove all files not in the subdirectory
        self.file_list = [file for file in self.file_list if file.startswith(sub_directory)]
        # Only load xls and xlsx files
        self.file_list = [file for file in self.file_list if file.endswith('.xls') or file.endswith('.xlsx')]
        self.sample_length = sample_length
        self.transform = transform

        # Fetch the min and max values of the glucose values
        self.min = float('inf')
        self.max = float('-inf')
        for file in self.file_list:
            glucose = self._fetch_raw_glucose_data(file)
            self.min = min(self.min, glucose.min().item())
            self.max = max(self.max, glucose.max().item())

        self.sample_interval = 15 * 60 # 15 minutes in seconds

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        glucose = self._fetch_raw_glucose_data(self.file_list[idx])
        normalized_glucose = (glucose - self.min) / (self.max - self.min)
        return normalized_glucose
        
    def _fetch_raw_glucose_data(self, filename: str):
        with self.zip_file.open(filename) as file:
            # Read the xlsx file
            df = pd.read_excel(file)
            glucose = torch.tensor(df.iloc[:, 1].values, dtype=torch.float32).unsqueeze(dim=-1).numpy()
            return glucose
        