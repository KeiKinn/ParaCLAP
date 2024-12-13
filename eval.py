import os
import torch
from hydra import compose, initialize
from transformers import logging
from wrapper import EvalWrapper
import librosa


if __name__ == '__main__':
    logging.set_verbosity_error()
    ckpt = torch.hub.load_state_dict_from_url(
            url="https://huggingface.co/KeiKinn/paraclap/resolve/main/best.pth.tar?download=true",
            map_location="cpu",
            check_hash=True,
        )

    with initialize(config_path="./configs"):
        cfg = compose(config_name="config")
    
    candidates = ['happy', 'sad', 'surprise', 'neutral']
    wavpath = ''
    waveform, sample_rate = librosa.load(file_path, sr=16000)
    x = torch.Tensor(waveform)

    candidate_tokens = tokenizer.batch_encode_plus(
        candidates,
        padding=True,
        truncation=True,
        return_tensors='pt'
    ).to(cfg.meta.device)

    model = CLAP(
        speech_name=cfg.models.speech,
        text_name=cfg.models.text,
        embedding_dim=768,
    )

    model.load_state_dict(ckpt, strict=False)
    model.to(cfg.meta.device)
    print(f'Checkpoint is loaded')
    model.eval()

    with torch.no_grad():
        z = model(
            x.squeeze(1).to(cfg.meta.device),
            candidate_tokens
        )
        similarity = compute_similarity(z[2], z[0], z[1])
        prediction = similarity.T.argmax(dim=1)
    
    result = candidates[prediction]

    print(result)