import os
from extend_path import  *
import audmetric
import matplotlib.pyplot as plt
import torch
import tqdm
import yaml
from hydra import compose, initialize
from omegaconf import (
    DictConfig
)
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay
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


def evaluate(cfg, model=None, tqdm_disable=False):
    print('Evaluation on TESS!')

    tokenizer = AutoTokenizer.from_pretrained(cfg.models.text)
    root = '[YOUR PATH]/tess'

    if model is None:
        model = CLAP(
            speech_name=cfg.models.speech,
            text_name=cfg.models.text,
            embedding_dim=768,
        )

        ckpt_path = os.path.join(cfg.meta.results, "best.pth.tar")
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path))
            print(f"Loaded checkpoint from {ckpt_path}")
            model.to(cfg.meta.device)
    else:
        print('Evaluate on training models')
    
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
    # save confusion-matrix
    cm = confusion_matrix(targets, predictions)
    disp = ConfusionMatrixDisplay(cm, display_labels=ds.emo_list)
    disp.plot()
    plt.savefig(f'temp/tess_ps_{slurm_id}.png')
    return results


def evaluate_test(cfg: DictConfig, slurm_job_id='43049') -> None:
    for idx in range(0, 3, 1):
        temp = os.path.join(cfg.meta.results, slurm_job_id + f'_{idx}')
        if os.path.exists(temp):
            print(f'evaluted on {slurm_job_id}_{idx}')
            break
    cfg.meta.results = temp
    evaluate(cfg, tqdm_disable=False)
