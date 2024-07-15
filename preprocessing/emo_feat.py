import sys
import os

# Get the current script's directory
script_dir = os.path.dirname(os.path.realpath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(script_dir)

sys.path.insert(0, parent_dir)
import torch
from dataset import TemplateDataset
import audformat
from models_xin import SpeechEncoder
import numpy as np

import tqdm


def get_ds(root_dir, emo_list, templates, fix_desc):
    db = audformat.Database.load(os.path.join(root_dir, "converted"))
    df_train = db["categories.consensus.train"].df
    df_train = df_train.reset_index()
    df_train["file"] = df_train["file"].apply(os.path.basename)
    df_train = df_train.set_index("file")
    df_train = df_train.loc[df_train["emotion"].isin(emo_list)]
    tr_ds = TemplateDataset(
        df_train,
        templates,
        os.path.join(root_dir, "original", "Audios"),
        fix_desc=fix_desc
    )
    return tr_ds
