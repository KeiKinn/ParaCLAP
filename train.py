# This script is used to train the model by Xin and Andreas
# Zero-shot emotion CLAP
# the text template is created by Andreas and picked randomly from the template pool
# the loss func is adapted from laion-clap clip loss

import os
import random
import time

import audformat
import audtorch
import numpy as np
import torch
import tqdm
from hydra import compose, initialize
from omegaconf import (
    OmegaConf
)
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from transformers import logging

from dataset import TemplateDataset
from evaluate import evaluate_msp
from evaluation.evaluate_iemo import evaluate
from loss_laion import ClipLoss
from models_xin import (
    CLAP
)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def print_nn(mm):
    def count_pars(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    num_pars = count_pars(mm)
    print(mm)
    print('# pars: {}'.format(num_pars))
    print('{} : {}'.format('device', device))


def print_flags(cfg):
    print('--------------------------- Flags -----------------------------')
    for flag in cfg.asdic():
        print('{} : {}'.format(flag, getattr(cfg, flag)))


def get_ds(args):
    db = audformat.Database.load(os.path.join(args.meta.dataset, "converted"))
    df_train = db["categories.consensus.train"].df
    df_train = df_train.reset_index()
    df_train["file"] = df_train["file"].apply(os.path.basename)
    df_train = df_train.set_index("file")
    # df_train = df_train.head(200)
    df_train = df_train.loc[df_train["emotion"].isin(args.data.emotions)]
    tr_ds = TemplateDataset(
        df_train,
        args.meta.templates,
        os.path.join(args.meta.dataset, "original", "Audios"),
        transform=audtorch.transforms.RandomCrop(5*16000, axis=-1),
        fix_desc=args.hparams.fix_desc
    )
    return tr_ds


def get_dl(args, tr_ds):
    tr_dl = torch.utils.data.DataLoader(
        dataset=tr_ds,
        batch_size=args.hparams.batch_size,
        shuffle=True,
        num_workers=0
    )
    print('Training set size: {}, iteration: {}'.format(len(tr_ds), len(tr_dl)))
    return tr_dl


def training_setting(args, model, param_groups=False):
    if param_groups:
        audio_params = [p for n, p in model.named_parameters() if 'audio_branch' in n]
        text_params = [p for n, p in model.named_parameters() if 'text_branch' in n]
        other_params = [p for n, p in model.named_parameters() if 'audio_branch' not in n and 'text_branch' not in n]
        print(f'audio params: {len(audio_params)}, text params: {len(text_params)},other params: {len(other_params)}')
        # freeze_branch_parameters(model.named_parameters(), 'text_branch', args.finetuning.freeze_text)
        # Define different parameter groups with different learning rates
        # {'params': audio_params, 'lr': args.hparams.learning_rate / 100},
        param_groups = [
            {'params': text_params, 'lr': args.hparams.learning_rate / 100},
            {'params': other_params, 'lr': args.hparams.learning_rate},
        ]
    else:
        param_groups = [
            {'params': model.parameters(), 'lr': args.hparams.learning_rate},
        ]

    optimizer = torch.optim.Adam(param_groups, lr=args.hparams.learning_rate)
    
    criterion = ClipLoss(mlp_loss=False)
    return optimizer, criterion


def train(
        loader,
        tokenizer,
        optimizer,
        device,
        writer,
        epoch,
        tqdm_disable
):
    model.train()
    total_loss = 0
    for index, (x, y, label) in tqdm.tqdm(
            enumerate(loader),
            total=len(loader),
            desc='Train',
            disable=tqdm_disable
    ):
        txt_token = tokenizer.batch_encode_plus(
                list(y),
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(device)
        # z: audio_feat, text_feat, logit_scale_a.exp
        z = model(x.squeeze(1).to(device), txt_token)

        #NOTE: laion-clip loss
        loss = criterion(*z)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if index % 100 == 0:
            print(f'{index} batch loss is {loss}')
            writer.add_scalar(
                'batch_loss',
                loss,
                global_step=epoch * len(loader) + index
            )
        total_loss += loss.data.cpu()
    print(f'total loss is {total_loss / len(loader)}')
    return total_loss / len(loader)


if __name__ == '__main__':
    machine, local = get_path_prefix()
    ckpt_folder = ''
    logging.set_verbosity_error()

    with initialize(config_path="configs"):
        args = compose(config_name="config")

    print(OmegaConf.to_yaml(args))
    args.meta.results = os.path.join(args.meta.results, ckpt_folder)
    experiment_folder = args.meta.results

    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
        os.makedirs(experiment_folder + '/ckpt')
        os.makedirs(experiment_folder + '/out')
        os.makedirs(experiment_folder + '/log')

    setup_seed(3407)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    tr_ds = get_ds(args)
    tr_dl = get_dl(args, tr_ds)

    model = CLAP(
        speech_name=args.models.speech,
        text_name=args.models.text,
        embedding_dim=768,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.models.text)

    model.to(device)
    print('-' * 64)

    optimizer, criterion = training_setting(args, model, args.hparams.param_groups)
    writer = SummaryWriter(log_dir=os.path.join(experiment_folder, 'log'))

    best_acc = 0.
    best_uar = 0.
    best_uar_epoch = 1

    re = evaluate_msp(args, model=model, tqdm_disable=True)
    for epoch in range(1, args.hparams.epochs + 1):
        start = time.time()
        train_loss = train(
            loader=tr_dl,
            optimizer=optimizer,
            tokenizer=tokenizer,
            writer=writer,
            epoch=epoch,
            device=device,
            tqdm_disable=args.meta.tqdm_disable
        )
        writer.add_scalar(
            'loss/train',
            train_loss,
            global_step=epoch
        )

        model.logit_scale.data.clamp_(0, 100)
        
        torch.save(model.state_dict(), os.path.join(
            experiment_folder, 'last.pth.tar'))
        print(f'[{epoch}] last model is saved!')

        re = evaluate_msp(args, model=model, tqdm_disable=True)
        if re['UAR'] > best_uar:
            best_uar = re['UAR']
            best_uar_epoch = epoch
            torch.save(model.state_dict(), os.path.join(
                experiment_folder, 'best.pth.tar'))
            print(f'[{epoch}] best model is updated!')
        if re['ACC'] > best_acc:
            best_acc = re['ACC']

        print(f'[{epoch}] best UAR for MSP is {best_uar} at epoch {best_uar_epoch}')

        writer.add_scalar(
            'eval/evaluation_msp',
            re['ACC'],
            global_step=epoch
        )
        writer.add_scalar(
            'eval/evaluation_msp_uar',
            re['UAR'],
            global_step=epoch
        )

        end = time.time()
        print('Total time for {}: {:.2f} hours'.format('train & test', (end - start) / 3600))

    writer.close()
