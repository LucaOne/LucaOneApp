#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/12/8 15:10
@project: LucaOneApp
@file: inference_embedding_dnabert2.py
@desc: inference embedding of DNABert2
'''

import os
import sys
import torch
import argparse
sys.path.append(".")
sys.path.append("..")
sys.path.append("../algorithms")
try:
    from .utils import available_gpu_id, calc_emb_filename_by_seq_id
    from .file_operator import fasta_reader, csv_reader
    from .llm.dnabert2.inference_embedding import predict_embedding
except ImportError:
    from algorithms.utils import available_gpu_id, calc_emb_filename_by_seq_id
    from algorithms.file_operator import fasta_reader, csv_reader
    from algorithms.llm.dnabert2.inference_embedding import predict_embedding


def get_args():
    parser = argparse.ArgumentParser(description='DNABert2 Embedding')
    parser.add_argument("--embedding_type", type=str, default="matrix", choices=["matrix", "vector"], help="the llm embedding type.")
    parser.add_argument("--trunc_type", type=str, default="right", choices=["left", "right"], help="llm trunc type.")
    parser.add_argument("--truncation_seq_length", type=int, default=4094, help="the llm truncation seq length(not contain [CLS] and [SEP].")
    parser.add_argument("--matrix_add_special_token", action="store_true", help="whether to add special token embedding vector in seq representation matrix")
    parser.add_argument("--input_file", type=str, default=None, help="the input filepath(.fasta or .csv)")
    parser.add_argument("--seq", type=str, default=None, help="when to input a seq")
    parser.add_argument("--seq_type", type=str, default=None, required=True, choices=["gene"], help="the input seq type")
    parser.add_argument("--save_path", type=str, default=None, help="embedding file save dir path")
    parser.add_argument("--id_idx", type=int, default=None, help="id col idx(0 start)")
    parser.add_argument("--seq_idx", type=int, default=None, help="seq col idx(0 start)")
    parser.add_argument("--embedding_complete",  action="store_true", help="when the seq len > inference_max_len, then the embedding matrix is completed by segment")
    parser.add_argument('--gpu', type=int, default=-1, help="the gpu id to use.")

    input_args = parser.parse_args()
    return input_args


def main(model_args):
    print(model_args)
    if model_args.gpu >= 0:
        gpu_id = model_args.gpu
    else:
        gpu_id = available_gpu_id()
        print("gpu_id: ", gpu_id)
    model_args.device = torch.device("cuda:%d" % gpu_id if gpu_id > -1 else "cpu")
    # model.to(model_args.device)
    assert model_args.input_file is not None or model_args.seq is not None
    print("input seq type: %s" % model_args.seq_type)
    print("args device: %s" % model_args.device)
    embedding_type = model_args.embedding_type
    save_path = model_args.save_path
    seq_type = model_args.seq_type
    # emb_save_path = os.path.join(save_path, "dnabert2", "117M")
    emb_save_path = save_path
    print("emb save dir: %s" % emb_save_path)
    if seq_type not in ["gene"]:
        seq_type = "gene"
    if not os.path.exists(emb_save_path):
        os.makedirs(emb_save_path)
    if model_args.input_file:
        done = 0
        file_reader = fasta_reader
        if model_args.input_file.endswith(".csv"):
            file_reader = csv_reader
        for row in file_reader(model_args.input_file):
            if model_args.id_idx is None or model_args.seq_idx is None:
                if len(row) > 2:
                    seq_id, seq = row[0].strip(), row[2].upper()
                else:
                    seq_id, seq = row[0].strip(), row[1].upper()
            else:
                seq_id, seq = row[model_args.id_idx].strip(), row[model_args.seq_idx].upper()
            '''
            if " " in seq_id:
                emb_filename = seq_id.split(" ")[0] + ".pt"
            else:
                emb_filename = seq_id + ".pt"
            if "/" in emb_filename:
                emb_filename = emb_filename.replace("/", "_")
            emb_filename = embedding_type + "_" + emb_filename
            '''
            emb_filename = calc_emb_filename_by_seq_id(seq_id=seq_id, embedding_type=embedding_type)
            embedding_filepath = os.path.join(emb_save_path, emb_filename)
            if not os.path.exists(embedding_filepath):
                ori_seq_len = len(seq)
                truncation_seq_length = model_args.truncation_seq_length
                if model_args.embedding_complete:
                    truncation_seq_length = ori_seq_len
                emb, processed_seq_len = predict_embedding([seq_id, seq_type, seq],
                                                           model_args.trunc_type,
                                                           embedding_type,
                                                           repr_layers=[-1],
                                                           truncation_seq_length=truncation_seq_length,
                                                           device=model_args.device,
                                                           version="dnabert2",
                                                           matrix_add_special_token=model_args.matrix_add_special_token
                                                           )
                while emb is None:
                    print("%s embedding error, max_len from %d truncate to %d" % (seq_id, truncation_seq_length, int(truncation_seq_length * 0.95)))
                    truncation_seq_length = int(truncation_seq_length * 0.95)
                    emb, processed_seq_len = predict_embedding([seq_id, seq_type, seq],
                                                               model_args.trunc_type,
                                                               embedding_type,
                                                               repr_layers=[-1],
                                                               truncation_seq_length=truncation_seq_length,
                                                               device=model_args.device,
                                                               version="dnabert2",
                                                               matrix_add_special_token=model_args.matrix_add_special_token
                                                               )
                # print("seq_len: %d" % len(seq))
                # print("emb shape:", embedding_info.shape)
                torch.save(emb, embedding_filepath)
            else:
                print("%s exists." % embedding_filepath)
            done += 1
            if done % 1000 == 0:
                print("embedding done: %d" % done)
        print("embedding over, done: %d" % done)
    elif model_args.seq:
        print("input seq length: %d" % len(model_args.seq))
        emb, processed_seq_len = predict_embedding(["input", model_args.seq],
                                                   model_args.trunc_type,
                                                   model_args.embedding_type,
                                                   repr_layers=[-1],
                                                   truncation_seq_length=model_args.truncation_seq_length,
                                                   device=model_args.device,
                                                   version="dnabert2",
                                                   matrix_add_special_token=model_args.matrix_add_special_token
                                                   )
        print(emb)


if __name__ == "__main__":
    input_args = get_args()
    main(input_args)

