import os

import audformat
import audmetric
import hydra
import numpy as np
import pandas as pd
import torch
import tqdm
import yaml
from omegaconf import (
    DictConfig
)
from transformers import AutoTokenizer

from dataset import (
    TargetDataset
)
from models_xin import (
    CLAP
)
from utils import (
    compute_similarity
)


def evaluate_msp(cfg, model=None, tqdm_disable=False):
    print('Evaluation on msp!')

    tokenizer = AutoTokenizer.from_pretrained(cfg.models.text)
    db = audformat.Database.load(os.path.join(cfg.meta.dataset, "converted"))

    df_test = db["categories.consensus.test1"].df
    df_test = df_test.reset_index()
    # Take the first several rows
    # df_test = df_test.head(4000)
    df_test["file"] = df_test["file"].apply(os.path.basename)
    df_test = df_test.set_index("file")

    candidates = list(df_test["emotion"].unique())
    # candidates_sentence = [f'this person is feeling {e}' for e in candidates]
    candidate_tokens = tokenizer.batch_encode_plus(
        candidates,
        padding=True,
        truncation=True,
        return_tensors='pt'
    ).to(cfg.meta.device)

    test_dataset = TargetDataset(
        df_test,
        target_column="emotion",
        wav_root=os.path.join(cfg.meta.dataset, "original", "Audios")
    )

    loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=1
    )
    if model is None:
        model = CLAP(
            speech_name=cfg.models.speech,
            text_name=cfg.models.text,
            embedding_dim=768,
        )

        ckpt_path = os.path.join(cfg.meta.results, "last.pth.tar")
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path))
            print(f"Loaded checkpoint from {ckpt_path}")
            model.to(cfg.meta.device)
    else:
        print('Evaluate on training models')

    model.eval()
    targets = []
    predictions = []
    for x, y in tqdm.tqdm(
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
            targets.append(y[0])
            predictions.append(candidates[prediction])
    results = {
        "ACC": audmetric.accuracy(targets, predictions),
        "UAR": audmetric.unweighted_average_recall(targets, predictions),
        "F1": audmetric.unweighted_average_fscore(targets, predictions)
    }
    print(yaml.dump(results))
    os.makedirs(cfg.meta.results, exist_ok=True)
    with open(os.path.join(cfg.meta.results, "results.msp.zsl.yaml"), "w") as fp:
        yaml.dump(results, fp)
    pd.DataFrame(
        data=np.stack([predictions, targets], axis=1),
        index=df_test.index,
        columns=["prediction", "emotion"]
    ).reset_index().to_csv(os.path.join(cfg.meta.results, "results.msp.zsl.csv"), index=False)
    torch.cuda.empty_cache()
    return results


@hydra.main(config_path="configs", config_name="config")
def evaluate_test(cfg: DictConfig, slurm_job_id='40732_0') -> None:
    print(f'evaluted on {slurm_job_id}')
    cfg.meta.results = os.path.join(cfg.meta.results, slurm_job_id)
    evaluate_msp(cfg, tqdm_disable=True)


if __name__ == "__main__":
    evaluate_test()
