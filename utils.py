import torch
import torch.nn.functional as F
import collections

def compute_similarity(logit_scale, audio_embeddings, text_embeddings):
    r"""Compute similarity between text and audio embeddings"""
    audio_embeddings = audio_embeddings/torch.norm(audio_embeddings, dim=-1, keepdim=True)
    text_embeddings = text_embeddings/torch.norm(text_embeddings, dim=-1, keepdim=True)

    similarity = logit_scale*text_embeddings @ audio_embeddings.T
    return similarity.T

def compute_logit(logit_scale, audio_embeddings, text_embeddings):
    logits_per_audio = logit_scale * audio_embeddings @ text_embeddings.T
    logits_per_text = logit_scale * text_embeddings @ audio_embeddings.T
    return logits_per_audio, logits_per_text

def laion_compute_similarity(logit_scale, audio_embeddings, text_embeddings):
    r"""Compute similarity between text and audio embeddings"""
    audio_embeddings = F.normalize(audio_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)

    similarity = logit_scale*audio_embeddings @ text_embeddings.T
    return similarity

def freeze_branch_parameters(named_parameters, branch_name, freeze_flag):
    branch_parameters = [
        p
        for n, p in named_parameters
        if branch_name in n
    ]
    if freeze_flag:
        print(f"Freezing {branch_name.capitalize()} parameters.")
        for param in branch_parameters:
            param.requires_grad = False

def format_emotion(emotion):
    if emotion == 'no_agreement':
        return 'there is no clear emotion.'
    else:
        return f'this person is feeling {emotion}.'


def preprocess_text(text_queries, tokenizer):
    r"""Load list of class labels and return tokenized text"""
    token_keys = ['input_ids', 'token_type_ids', 'attention_mask']
    tokenized_texts = []
    for ttext in text_queries:
        tok = tokenizer.encode_plus(
            text=ttext, add_special_tokens=True, max_length=77, padding='max_length', return_tensors="pt")
        for key in token_keys:
            tok[key] = tok[key].reshape(-1).cuda()
        tokenized_texts.append(tok)
    return default_collate(tokenized_texts)

def default_collate(batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(
                        default_collate_err_msg_format.format(elem.dtype))

                return default_collate([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, collections.abc.Mapping):
            return {key: default_collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(default_collate(samples) for samples in zip(*batch)))
        elif isinstance(elem, collections.abc.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError(
                    'each element in list of batch should be of equal size')
            transposed = zip(*batch)
            return [default_collate(samples) for samples in transposed]

        raise TypeError(default_collate_err_msg_format.format(elem_type))