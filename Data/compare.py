import glob
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import librosa

class ComParE2012_Likability(Dataset):
    def __init__(
        self,
        path: str
    ):
        self.path = path
        
        df = pd.read_csv(
            os.path.join(
                self.path,
                "LSC_final_geheim.csv"
            ),
            sep=";"
        )
        df["split"] = df["file"].apply(lambda x: x.split("_")[0])
        df = df.set_index("file")
        self.df = df.loc[df["split"] == "test"]

        self.emotion_map = {
            "L": 0,
            "NL": 1
        }
        self.emo_list = [
            "likeable",
            "not likeable"
        ]
    def __getitem__(self, idx):
        file_name = self.df.index[idx]
        file_path = os.path.join(self.path, "wav", file_name)

        # Load the audio file
        waveform, sample_rate = librosa.load(file_path, sr=16000)
        emotion = self.df.loc[file_name, "likability"]

        return torch.Tensor(waveform), emotion, self.emotion_map[emotion]

    def __len__(self):
        return len(self.df)


class ComParE2011_Intoxication(Dataset):
    def __init__(
        self,
        path: str
    ):
        self.path = path
        
        df = pd.read_csv(
            os.path.join(
                self.path,
                "ALC.csv"
            ),
        )
        df["split"] = df["file"].apply(lambda x: x.split("_")[0])
        df = df.loc[df["split"] == "TE"]
        self.df = df.set_index("file")

        self.emotion_map = {
            "AL": 0,
            "NAL": 1
        }
        self.emo_list = [
            "intoxicated with alcohol",
            "not intoxicated with alcohol"
        ]
    
    
    def __getitem__(self, idx):
        file_name = self.df.index[idx]
        file_path = os.path.join(self.path, "DIST", "TEST", file_name + ".wav")

        # Load the audio file
        waveform, sample_rate = librosa.load(file_path, sr=16000)
        emotion = self.df.loc[file_name, "challenge_class"]

        return torch.Tensor(waveform), emotion, self.emotion_map[emotion]

    def __len__(self):
        return len(self.df)


