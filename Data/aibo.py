import glob
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import librosa

class AIBO(Dataset):
    def __init__(
        self,
        path: str,
        task: str = "2cl"  # choices: 2cl, 5cl
    ):
        self.path = path
        self.task = task
        df = pd.read_csv(
            os.path.join(
                self.path,
                "labels",
                "IS2009EmotionChallenge",
                f"chunk_labels_{self.task}_corpus.txt"
            ),
            header=None,
            sep=" "
        )
        df = df.rename(
            columns={
                0: "id",
                1: "class",
                2: "conf"
            }
        )
        df["file"] = df["id"].apply(lambda x: x + ".wav")
        df["school"] = df["id"].apply(lambda x: x.split("_")[0])
        df["speaker"] = df["id"].apply(lambda x: x.split("_")[1])
        df = df.set_index("file")
        # only need the test set
        self.df = df.loc[df["school"] == "Mont"]

        self.emotion_map = {
            "2cl": {
                "IDL": 0,
                "NEG": 1
            },
            "5cl": {
                "A": 0,
                "E": 1,
                "N": 2,
                "P": 3,
                "R": 4
            }
        }
        emo_lis_t = {
                "2cl": [
                    "non-negative",
                    "negative"
                ],
                "5cl": [
                    "angry",
                    "emphatic",
                    "neutral",
                    "motherese",
                    "rest"
                ]
            }
        self.emo_list = emo_lis_t[self.task]
    
    def __getitem__(self, idx):
        file_name = self.df.index[idx]
        file_path = os.path.join(self.path, "wav", file_name)

        # Load the audio file
        waveform, sample_rate = librosa.load(file_path, sr=16000)
        emotion = self.df.loc[file_name, "class"]

        return torch.Tensor(waveform), emotion, self.emotion_map[self.task][emotion]

    def __len__(self):
        return len(self.df)
