import os

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

from Data.iemocap import (
    IEMOCAP
)
from models_xin import (
    CLAP
)
from utils import (
    compute_similarity, format_emotion
)


def evaluate(cfg, model=None, tqdm_disable=False, slurm_id=None):
    print('Evaluation on IEMOCAP!')

    tokenizer = AutoTokenizer.from_pretrained(cfg.models.text)
    root = '[YOUR PATH]'

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

    candidates_ = ['neutral', 'happy', 'anger', 'sadness']
    candidates = [format_emotion(emo) for emo in candidates_]
    candidate_tokens = tokenizer.batch_encode_plus(
        candidates,
        padding=True,
        truncation=True,
        return_tensors='pt'
    ).to(cfg.meta.device)

    re = []
    cm_tgt = []
    cm_pre = []

    sessions = [1, 2, 3, 4, 5]
    for k_fold in range(5):
        print(f'Fold {k_fold}')
        val_session = [sessions[k_fold]]
        ds = IEMOCAP(root=root,
                         sessions=val_session)
        loader = torch.utils.data.DataLoader(
            dataset=ds,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )

        model.eval()
        targets = []
        predictions = []
        for x, text, y in tqdm.tqdm(
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
        re.append(results)
        print(f'result at fold {k_fold} are {yaml.dump(results)}')
        torch.cuda.empty_cache()
        cm_tgt.extend(targets)
        cm_pre.extend(predictions)
    # get average acc uar and f1
    for k in re[0].keys():
        results[k] = sum([re[i][k] for i in range(5)]) / 5
    print(f'Final:\n{yaml.dump(results)}')

    # save confusion-matrix
    cm = confusion_matrix(cm_tgt, cm_pre)
    disp = ConfusionMatrixDisplay(cm, display_labels=candidates_)
    disp.plot()
    plt.savefig(f'temp/iemo_{slurm_id}.png')
    return results


def evaluate_test(cfg: DictConfig, slurm_job_id='44825') -> None:
    for idx in range(0, 3, 1):
        temp = os.path.join(cfg.meta.results, slurm_job_id + f'_{idx}')
        if os.path.exists(temp):
            print(f'evaluted on {slurm_job_id}_{idx}')
            break
    cfg.meta.results = temp
    evaluate(cfg, tqdm_disable=False)
