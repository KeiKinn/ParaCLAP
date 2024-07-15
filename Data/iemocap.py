import torch
import torchaudio

import audtorch

from torchaudio.datasets.utils import _load_waveform


class IEMOCAP(torchaudio.datasets.IEMOCAP):
    def __init__(self, transform=None, exlude_list=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if exlude_list is None:
            exlude_list = ['fru']
        self.transform = transform
        self.exlude_list = exlude_list
        self.data_ = []
        self.filter_data()

    def __getitem__(self, n):
        """
        Args:
            n (int): Index of the item in the dataset
        Returns:
            tuple: (waveform, sample_rate, label, speaker_id, utterance_id, extra_info)
        """
        data = self.get_item(n)
        if self.transform is not None:
            # self.transform.__setattr__('idx', [0, self.transform.size])
            data_ = self.transform(data[0])
            # check data_ is not a tensor
            if not isinstance(data_, torch.Tensor):
                data_ = torch.tensor(data_)
            data[0] = data_
        wav, text, label = self.format_data(data)
        return wav, text, label

    def get_item(self, idx):
        temp = self.data_[idx]
        wav = _load_waveform(self._path, temp[0], temp[1])
        temp = (wav,) + temp[1:]
        data = list(temp)
        return data

    def filter_data(self):
        rn = super().__len__()
        for idx in range(0, rn):
            temp = super().get_metadata(idx)
            if temp[3] not in self.exlude_list:
                self.data_.append(temp)

    def format_data(self, data):
        emo_dict = {'neu': 'neutral',
                    'hap': 'happy',
                    'ang': 'angry',
                    'sad': 'sad',
                    'exc': 'happy',
                    'fru': 'frustrated'}
        label_dict = {'neu': 0,
                      'hap': 1,
                      'ang': 2,
                      'sad': 3,
                      'exc': 1,
                      'fru': 4}
        wav = data[0]
        label = data[3]
        text = f'Emotion is {emo_dict[label]}'
        return wav, text, label_dict[label]

    def __len__(self):
        return len(self.data_)


class IEMOCAP_G(IEMOCAP):
    def format_data(self, data):
        label_dict = {'neu': 0,
                      'hap': 1,
                      'ang': 2,
                      'sad': 3,
                      'exc': 1,
                      'fru': 5}
        wav = data[0]
        label = data[3]
        gender = 0 if data[4][-1] == 'M' else 1
        return wav, gender, label_dict[label]

