import os
import audmetric
import torch
import tqdm
import yaml
from hydra import compose, initialize
from omegaconf import (
    DictConfig
)
from transformers import AutoTokenizer
from transformers import logging

from Data.tess import (
    TESS as Ds
)
from models_xin import (
    CLAP
)
from utils import (
    compute_similarity
)


def evaluate(cfg, root=None, ckpt=None, tqdm_disable=False):
    print('Evaluation on TESS!')

    tokenizer = AutoTokenizer.from_pretrained(cfg.models.text)

    model = CLAP(
        speech_name=cfg.models.speech,
        text_name=cfg.models.text,
        embedding_dim=768,
    )

    model.load_state_dict(ckpt, strict=False)
    model.to(cfg.meta.device)
    print(f'Checkpoint is loaded')
    
    ds = Ds(dataset_dir=root)

    candidates = ds.emo_list
    # candidates = [format_emotion(emo) for emo in candidates]
    candidate_tokens = tokenizer.batch_encode_plus(
        candidates,
        padding=True,
        truncation=True,
        return_tensors='pt'
    ).to(cfg.meta.device)

    loader = torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    model.eval()
    targets = []
    predictions = []
    for x, _, y in tqdm.tqdm(
            loader,
            total=len(loader),
            desc="Evaluate",
            disable=tqdm_disable
    ):
        with torch.no_grad():
            z = model(
                x.squeeze(1).to(cfg.meta.device),
                candidate_tokens
            )
            similarity = compute_similarity(z[2], z[0], z[1])
            prediction = similarity.T.argmax(dim=1)
            targets.append(y[0].item())
            predictions.append(prediction.item())
    results = {
        "ACC": audmetric.accuracy(targets, predictions),
        "UAR": audmetric.unweighted_average_recall(targets, predictions),
        "F1": audmetric.unweighted_average_fscore(targets, predictions)
    }
    print(f'result are {yaml.dump(results)}')
    torch.cuda.empty_cache()
    return results


def evaluate_test(cfg: DictConfig, ckpt_path=None) -> None:
    cfg.meta.ckpt_path = ckpt_path 
    evaluate(cfg, tqdm_disable=False)