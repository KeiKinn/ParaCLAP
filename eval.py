import os
import torch
import librosa
from transformers import logging
from transformers import AutoTokenizer
from models_xin import CLAP
from utils import compute_similarity


if __name__ == '__main__':
    logging.set_verbosity_error()
    ckpt = torch.hub.load_state_dict_from_url(
            url="https://huggingface.co/KeiKinn/paraclap/resolve/main/best.pth.tar?download=true",
            map_location="cpu",
            check_hash=True,
        )
    
    text_model = 'bert-base-uncased'
    audio_model = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    candidates = ['happy', 'sad', 'surprise', 'angry'] # free to adapt it to your need
    wavpath = '[Waveform path]' # single channel wavform

    waveform, sample_rate = librosa.load(wavpath, sr=16000)
    x = torch.Tensor(waveform)

    tokenizer = AutoTokenizer.from_pretrained(text_model)

    candidate_tokens = tokenizer.batch_encode_plus(
        candidates,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    model = CLAP(
        speech_name=audio_model,
        text_name=text_model,
        embedding_dim=768,
    )

    model.load_state_dict(ckpt)
    model.to(device)
    print(f'Checkpoint is loaded')
    model.eval()

    with torch.no_grad():
        z = model(
            x.unsqueeze(0).to(device),
            candidate_tokens
        )

    similarity = compute_similarity(z[2], z[0], z[1])
    prediction = similarity.T.argmax(dim=1)
    
    result = candidates[prediction]

    print(result)