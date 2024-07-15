import os
import torch
from torch.utils.data import Dataset
import librosa

class CREMAD(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = os.path.join(dataset_dir, 'AudioWAV')
        self.fl = self.get_meta()
        self.emo_dict = {"SAD":0, "ANG": 1, "DIS":2, "FEA":3, "HAP":4, "NEU":5}
        self.emo_list = ['sad', 'angry', 'disgust', 'fear','happy', 'neutral']
    
    def __getitem__(self, idx):
        file_name = self.fl[idx]
        file_path = os.path.join(self.dataset_dir, file_name)

        # Load the audio file
        waveform, sample_rate = librosa.load(file_path, sr=16000)

        # Extract emotion from the file name
        _, _, emotion, _ = file_name.split('_')

        return torch.Tensor(waveform).unsqueeze(0), emotion, self.emo_dict[emotion]

    def __len__(self):
        return len(self.fl)

    def get_meta(self):
        assert os.path.exists(self.dataset_dir) , 'Cannot find the dataset!'
        file_list = [file for file in os.listdir(self.dataset_dir) if file.endswith('.wav')]
        return file_list

