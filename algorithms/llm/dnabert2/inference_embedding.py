#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.**@**.com
@tel: 137****6540
@datetime: 2023/3/20 13:23
@project: LucaOneApp
@file: inference_embedding.py
@desc: inference embedding using DNABert2
'''

import sys
import torch
sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../algorithms")
try:
    from file_operator import fasta_reader
    from utils import clean_seq_luca
except ImportError:
    from algorithms.file_operator import fasta_reader
    from algorithms.utils import clean_seq_luca
from transformers import AutoTokenizer, AutoModel

model_id = 'zhihan1996/DNABERT-2-117M'

dnabert2_global_model, dnabert2_global_alphabet, dnabert2_global_version = None, None, None


def predict_embedding(
        sample,
        trunc_type,
        embedding_type,
        repr_layers=[-1],
        truncation_seq_length=4094,
        device=None,
        version="dnabert2",
        matrix_add_special_token=False,
        save_type="numpy"
):
    """
    use sequence to predict the seq embedding matrix or vector([CLS])
    :param sample: [seq_id, seq]
    :param trunc_type: right or left when the input seq is too longer
    :param embedding_type: [CLS] vector or embedding matrix
    :param repr_layers: [-1], the last layer
    :param truncation_seq_length: such as: [4094, 2046, 1982, 1790, 1534, 1278, 1150, 1022]
    :param device: running device
    :param version: llm version
    :param matrix_add_special_token: embedding matrix contains [CLS] and [SEP] vector or not
    :param save_type:
    :return: embedding, processed_seq_len
    """

    global dnabert2_global_model, dnabert2_global_alphabet, dnabert2_global_version
    assert "bos" in embedding_type or "representations" in embedding_type \
           or "matrix" in embedding_type or "vector" in embedding_type or "contacts" in embedding_type
    if len(sample) > 2:
        seq_id, seq = sample[0], sample[2]
    else:
        seq_id, seq = sample[0], sample[1]
    processed_seq = clean_seq_luca(seq_id, seq)
    if len(processed_seq) > truncation_seq_length:
        if trunc_type == "left":
            processed_seq = processed_seq[-truncation_seq_length:]
        else:
            processed_seq = processed_seq[:truncation_seq_length]
    if dnabert2_global_model is None or dnabert2_global_alphabet is None or dnabert2_global_version is None or dnabert2_global_version != version:
        if version == "dnabert2":
            dnabert2_global_alphabet = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            dnabert2_global_model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        else:
            raise Exception("not support this version=%s" % version)
        dnabert2_global_version = version

    '''
    if torch.cuda.is_available() and device is not None:
        dnabert2_global_model = dnabert2_global_model.to(device)
    elif torch.cuda.is_available():
        dnabert2_global_model = dnabert2_global_model.cuda()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("llm use cpu")
    '''
    if device is None:
        device = next(dnabert2_global_model.parameters()).device
    else:
        model_device = next(dnabert2_global_model.parameters()).device
        if device != model_device:
            dnabert2_global_model = dnabert2_global_model.to(device)

    dnabert2_global_model.eval()

    inputs = dnabert2_global_alphabet(processed_seq, return_tensors='pt')["input_ids"]
    embeddings = {}
    with torch.no_grad():
        if torch.cuda.is_available():
            inputs = inputs.to(device=device, non_blocking=True)
        try:
            out = dnabert2_global_model(inputs)
            # inputs contain [CLS] in head and [SEP] in tail
            truncate_len = min(truncation_seq_length, inputs.shape[1] - 2)
            processed_seq_len = truncate_len + 2
            if "representations" in embedding_type or "matrix" in embedding_type:
                # embedding matrix contain [CLS] and [SEP] vector
                if matrix_add_special_token:
                    embedding = out[0].to(device="cpu")[0, 0: truncate_len + 2].clone()
                else:
                    embedding = out[0].to(device="cpu")[0, 1: truncate_len + 1].clone()
                if save_type == "numpy":
                    embedding = embedding.numpy()
                embeddings["representations"] = embedding
            if "bos" in embedding_type or "vector" in embedding_type:
                embedding = out[0].to(device="cpu")[0, 0].clone()
                if save_type == "numpy":
                    embedding = embedding.numpy()
                embeddings["bos_representations"] = embedding
            if "contacts" in embedding_type:
                # to do
                embeddings["contacts"] = None
            if len(embeddings) > 1:
                return embeddings, processed_seq_len
            elif len(embeddings) == 1:
                return list(embeddings.items())[0][1], processed_seq_len
            else:
                return None, None
        except RuntimeError as e:
            if e.args[0].startswith("CUDA out of memory"):
                print(f"Failed (CUDA out of memory) on sequence {seq_id} of length {len(seq)}.")
                print("Please reduce the 'truncation_seq_length'")
            else:
                print(f"sequence {seq_id} of length {len(seq)}.")
                print(e)
            raise Exception(e)
    return None, None
