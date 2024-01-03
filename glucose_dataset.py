from zipfile import ZipFile
from torch.utils.data import Dataset
import pandas as pd
import torch

class GlucoseDataset(Dataset):
    def __init__(self, zip_file_path, sub_directory, input_length=16, output_length=1, transform=None):
        self.sub_directory = sub_directory
        self.zip_file = ZipFile(zip_file_path, 'r')
        self.file_list = self.zip_file.namelist()  # Get list of files in the zip
        # Remove all files not in the subdirectory
        self.file_list = [file for file in self.file_list if file.startswith(sub_directory)]
        # Only load xls and xlsx files
        self.file_list = [file for file in self.file_list if file.endswith('.xls') or file.endswith('.xlsx')]
        self.input_length = input_length
        self.output_length = output_length
        self.transform = transform

        # Sample data
        self.sample_interval: float = 0.25 * 60 * 60 # 15 minutes in seconds
        samples_raw: list[torch.Tensor] = []

        # Fetch the min and max values of the glucose values
        for file in self.file_list:
            glucose = self._fetch_raw_glucose_data(file)
            for i in range(0, len(glucose) - (self.input_length + self.output_length + 1)):
                new_sample = glucose[i:i+self.input_length+self.output_length]
                samples_raw.append(new_sample)
        self.samples = torch.stack(samples_raw)

        # Normalze the data
        self.min = torch.min(self.samples)
        self.max = torch.max(self.samples)
        self.samples = (self.samples - self.min) / (self.max - self.min)

        # Shuffle the samples along the first dimension
        shuffled_indices = torch.randperm(self.samples.size(0))
        self.samples = self.samples[shuffled_indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        return sample[:self.input_length], sample[-1]
        
    def _fetch_raw_glucose_data(self, filename: str) -> torch.Tensor:
        with self.zip_file.open(filename) as file:
            # Read the xlsx file
            df = pd.read_excel(file)
            glucose = torch.tensor(df.iloc[:, 1].values, dtype=torch.float32).unsqueeze(dim=-1)
            return glucose
        