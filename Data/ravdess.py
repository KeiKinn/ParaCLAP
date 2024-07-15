
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
import glob
import librosa
import numpy as np
import pandas as pd

class RAVDESS(Dataset):
    def __init__(self, dataset_dir, speech_list= None, out_dir = None):
        self.dataset_dir = dataset_dir
        self.out_dir = out_dir
        
        if speech_list == None:
            self.speech_list = self.get_meta().speech_list.tolist()
        else:
            self.speech_list = speech_list

        self.emo_list = ["neutral","calm","happy","sad","angry","fearful","disgust","surprised"]
    
    def __getitem__(self, index):
        fpath = self.speech_list[index]
        fn = fpath.split("/")[-1].split(".")[0]
        
        sig, sr = librosa.load(self.speech_list[index], sr=16000)
            
        features = ['modality', 'vocal_channel', 'emotion', 'emotion_intensity', 'statement', 'repetition', 'actor']
        info = {f:c for c, f in zip(fn.split("-"), features)}
        emotion = emotion_dict[info['emotion']]
        return torch.Tensor(sig).unsqueeze(0), emotion, int(info['emotion']) - 1
    
    def __len__(self):
        return len(self.speech_list)

    def get_meta(self):
        speech_list = glob.glob(os.path.join(self.dataset_dir, '*/*.wav'))
        features = ['modality', 'vocal_channel', 'emotion', 'emotion_intensity', 'statement', 'repetition', 'actor']
        meta_info = pd.DataFrame(columns=features)
        for fpath in speech_list:
            fn = fpath.split("/")[-1].split(".")[0]
            
            speech_info = {f:[c] for c, f in zip(fn.split("-"), features)}
            meta_info = pd.concat( (meta_info, pd.DataFrame(speech_info)), axis=0)
        meta_info.insert(0, "speech_list", speech_list)
        return meta_info

emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
 }

