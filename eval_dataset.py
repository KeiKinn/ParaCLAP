import os
import torch
from hydra import compose, initialize
from transformers import logging
from wrapper import EvalWrapper


if __name__ == '__main__':
    logging.set_verbosity_error()
    ckpt = torch.hub.load_state_dict_from_url(
            url="https://huggingface.co/KeiKinn/paraclap/resolve/main/best.pth.tar?download=true",
            map_location="cpu",
            check_hash=True,
        )
    with initialize(config_path="./configs"):
        cfg = compose(config_name="config")
    dataset='tess'
    data_root = '[Path to the dataset]'
    _, evaluate = EvalWrapper(dataset).set_eval()
    results = evaluate(cfg, data_root, ckpt=ckpt)