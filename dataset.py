import audiofile
import json
import numpy as np
import os
import random
import torch


emotion_dict = {
    'neutral': 0,
    'happiness': 1,
    'emotion': 2,
    'other': 3,
    'surprise': 4,
    'fear': 5,
    'contempt': 6,
    'disgust': 7,
    'sadness': 8,
    'anger': 9
}

def filter_texts(text_list, keywords):
    """
    Remove texts from the list that contain any of the specified keywords.

    Args:
        text_list (list): List of texts.
        keywords (list): List of keywords to filter.

    Returns:
        list: Filtered list of texts.
    """
    filtered_list = [text for text in text_list if all(keyword not in text for keyword in keywords)]
    return filtered_list


class TemplateDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            df,
            template_root,
            wav_root,
            transform=None,
            fix_desc=False
    ):
        self.df = df
        self.template_root = template_root
        self.wav_root = wav_root
        self.max_templates = 5
        self.p = np.linspace(1, 0, self.max_templates)
        self.p /= self.p.sum()
        self.transform = transform
        self.fix_desc = fix_desc

        print(f'mx te {self.max_templates}, fix {self.fix_desc}')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        file = self.df.index[item]
        audio, fs = audiofile.read(
            os.path.join(self.wav_root, file),
            always_2d=True
        )
        audio = audio.mean(0, keepdims=True)
        with open(os.path.join(self.template_root, f"{file}.json"), "r") as fp:
            templates = json.load(fp)

        emo_p = [t for t in templates if 'emotion' in t][0]
        emo = emotion_dict[emo_p.split(' ')[-1]]

        # templates = filter_texts(templates, keywords=list(emotion_dict.keys()))
        # N = np.random.choice(range(1, self.max_templates + 1), p=self.p)
        # text = " and ".join(random.sample(templates, N))
        # text = text + " and " + emo_p
        text ='speaker '+' and '.join(templates)
        if self.fix_desc:
            text = emo_p
        if self.transform is not None:
            audio = self.transform(audio)
        return audio, text, emo, file


class TargetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df,
        target_column,
        wav_root,
        transform = None
    ):
        self.df = df
        self.target_column = target_column
        self.wav_root = wav_root
        self.transform = transform
        
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        file = self.df.index[item]
        audio, fs = audiofile.read(
            os.path.join(self.wav_root, file),
            always_2d=True
        )
        audio = audio.mean(0, keepdims=True)
        if self.transform is not None:
            audio = self.transform(audio)
        return audio, self.df.loc[file, self.target_column]


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generate templates")
    parser.add_argument("-dataset", "-d")
    parser.add_argument("-features", "-f")
    parser.add_argument("-templates", "-t")
    args = parser.parse_args()

    db = audformat.Database.load(os.path.join(args.dataset, "converted"))
    df = db["categories.consensus.train"].df
    df = pd.concat((df, db["dimensions.consensus.train"].get(index=df.index)), axis=1)
    df = pd.concat((df, db["speaker"].get(index=df.index)), axis=1)
    df["gender"] = df["speaker"].apply(lambda x: db.schemes["speaker"].labels[x]["gender"])
    features = pd.read_csv(args.features).set_index("file")

    features = features.reindex(df.index)
    df = pd.concat((df, features), axis=1)
    df = df.reset_index()
    df["file"] = df["file"].apply(os.path.basename)
    df = df.set_index("file")

    dataset = TemplateDataset(df[:100], args.templates, os.path.join(args.dataset, "original", "Audios"))

    start = time.time()
    for x, y in tqdm.tqdm(
        dataset,
        total=len(dataset),
        desc='Iterating'
    ):
        pass
    print(y)
    end = time.time()
    print(f"{end-start} seconds elapsed")
    # Full dataset: 0.3435
    # Only text: 0.0472
    # Only audio: 0.3006
