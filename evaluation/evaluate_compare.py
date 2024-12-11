import os
from extend_path import  *
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

from Data.compare import (
    ComParE2011_Intoxication as Ds
)
from models_xin import (
    CLAP
)
from utils import (
    compute_similarity
)


def evaluate(cfg, model=None, tqdm_disable=False):
    print('Evaluation on compare 11!')

    tokenizer = AutoTokenizer.from_pretrained(cfg.models.text)
    root = '[YOUR PATH]/IS2011ALCCHALLENGE/'

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
    
    ds = Ds(path=root)

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


if __name__ == '__main__':
    logging.set_verbosity_error()
    ckpt_path = os.path.join(os.getcwd(), 'ckpt/best.pth.tar')
    with initialize(config_path="../configs"):
        cfg = compose(config_name="config")
    
    cfg.meta.ckpt_path = ckpt_path
    evaluate(cfg)
