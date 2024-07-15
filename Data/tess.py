import os
import torch
from torch.utils.data import Dataset
import librosa

class TESS(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = os.path.join(dataset_dir, 'data')
        self.fl = self.get_meta()
        self.emo_list = ['neutral', 'happy', 'sadness', 'angry', 'fear', 'disgust', 'pleasant surprise']
        self.emo_dict = {'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3,'fear': 4, 'disgust': 5,'ps': 6}
    
    def __getitem__(self, idx):
        file_name = self.fl[idx]
        file_path = os.path.join(self.dataset_dir, file_name)

        # Load the audio file
        waveform, sample_rate = librosa.load(file_path, sr=16000)

        # Extract emotion from the file name
        _, _, emotion = file_name.split('_')
        emotion = emotion.split('.')[0]

        return torch.Tensor(waveform).unsqueeze(0), emotion, self.emo_dict[emotion]

    def __len__(self):
        return len(self.fl)

    def get_meta(self):
        assert os.path.exists(self.dataset_dir) , 'Cannot find the dataset!'
        file_list = [file for file in os.listdir(self.dataset_dir) if file.endswith('.wav')]
        return file_list
