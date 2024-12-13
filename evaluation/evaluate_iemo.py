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

from Data.iemocap import (
    IEMOCAP
)
from models_xin import (
    CLAP
)
from utils import (
    compute_similarity, format_emotion
)


def evaluate(cfg, root=None, ckpt=None, tqdm_disable=False, slurm_id=None):
    print('Evaluation on IEMOCAP!')

    tokenizer = AutoTokenizer.from_pretrained(cfg.models.text)

    model = CLAP(
        speech_name=cfg.models.speech,
        text_name=cfg.models.text,
        embedding_dim=768,
    )

    model.load_state_dict(ckpt, strict=False)
    model.to(cfg.meta.device)
    print(f'Checkpoint is loaded')

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

    return results