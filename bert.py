# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import argparse
import io
import json
import logging
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

import senteval
from pytorch_pretrained_bert.modeling import (
    BertForMaskedLM,
    BertModel,
    global_states,
)
from pytorch_pretrained_bert.tokenization import BertTokenizer

# Set PATHs
# PATH_TO_SENTEVAL = "../"
PATH_TO_DATA = "/home/renxuancheng/SentEval-Data"

BERT_MODELS = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-base-multilingual",
    "bert-base-chinese",
]


# METHOD = "mean"

# import SentEval
# sys.path.insert(0, PATH_TO_SENTEVAL)


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, choices=BERT_MODELS)
    # parser.add_argument("bert_layer", type=int)
    parser.add_argument(
        "method", type=str, choices=["mean", "first", "max", "last"]
    )
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/home/renxuancheng/pytorch-pretrained-BERT/pretrained_bert",
    )
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    args.TOKENIZER_PATH_TEMPLATE = os.path.join(args.model_dir, "{}-vocab.txt")
    args.MODEL_PATH_TEMPLATE = os.path.join(args.model_dir, "{}.tar.gz")

    return args

    # METHOD = args.method

    # CACHE_PATH = os.path.join(os.path.dirname(__file__), args.cache_dir)
    # os.makedirs(CACHE_PATH, exist_ok=True)

    # BERT_MODEL = args.bert_model
    # BERT_LAYER = None  # args.bert_layer

    # USE_CUDA = args.cuda
    # USE_BATCH = args.batch


# class CacheList:
#     def __init__(self):

#         self.cache_path = os.path.join(
#             CACHE_PATH, f"{BERT_MODEL}.cache_list.json"
#         )
#         self.cache_pattern = os.path.join(
#             CACHE_PATH, f"{BERT_MODEL}_ex_{{}}.pt.tar"
#         )

#         if os.path.exists(self.cache_path):
#             with open(self.cache_path, "r", encoding="utf8") as f:
#                 self.cache_list = json.load(f)
#         else:
#             self.cache_list = {}

#     def add(self, example, data):

#         if example in self.cache_list:
#             return

#         example_id = len(self.cache_list)

#         torch.save(data, self.cache_pattern.format(example_id))
#         self.cache_list[example] = example_id
#         self._dump()

#     def get(self, example):
#         example_id = self.cache_list[example]
#         return torch.load(
#             self.cache_pattern.format(example_id), map_location="cpu"
#         )

#     def _dump(self):
#         with open(self.cache_path, "w", encoding="utf8") as f:
#             json.dump(self.cache_list, f)


def get_hidden_batch(params, batch, func=lambda x: torch.mean(x, dim=0)):

    device = "cuda" if params.args.cuda else "cpu"

    batch_ids = []
    lens = []
    for example in batch:
        # print(example)
        example = " ".join(example)
        tokens = params.tokenizer.tokenize(example)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        ids = params.tokenizer.convert_tokens_to_ids(tokens)
        batch_ids.append(ids)
        lens.append(len(tokens))

    max_len = max(lens)
    batch_size = len(batch)
    # logging.info("%d %s", batch_size, str(max_len))
    with torch.no_grad():
        id_t = torch.zeros(
            (batch_size, max_len), dtype=torch.long, device=device
        )
        mask_t = torch.zeros(
            (batch_size, max_len), dtype=torch.long, device=device
        )
        type_t = torch.ones(
            (batch_size, max_len), dtype=torch.long, device=device
        )

        for i, ids in enumerate(batch_ids):
            id_t[i, : len(ids)] = torch.tensor(
                ids, dtype=torch.long, device=device
            )
            mask_t[i, : len(ids)] = 1

        _ = params.bert(id_t, token_type_ids=type_t, attention_mask=mask_t)

        sent_hiddens = global_states["hidden"][
            params.args.layer
        ]  # [batch, len, hidden]
        global_states["hidden"] = []

        results = []
        for i, seq_len in enumerate(lens):
            result = func(sent_hiddens[i, :seq_len])
            # logging.info("%s", str(result))
            results.append(result)
        results = torch.stack(results, dim=0).cpu().numpy()

    return results


def get_hidden(params, batch, func=lambda x: torch.mean(x, dim=0)):
    """
    func: [len, hidden] -> [hidden]
    """
    hiddens = []
    with torch.no_grad():
        for example in batch:

            example = " ".join(example)
            tokens = params.tokenizer.tokenize(example)
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            ids = params.tokenizer.convert_tokens_to_ids(tokens)
            types = [0] * len(ids)
            mask = [1] * len(ids)

            id_t = torch.tensor([ids], dtype=torch.long)
            mask_t = torch.tensor([mask], dtype=torch.long)
            type_t = torch.tensor([types], dtype=torch.long)

            _ = params.bert(id_t, token_type_ids=type_t, attention_mask=mask_t)

            sent_hiddens = global_states["hidden"][params.args.layer][
                0
            ]  # [len, hidden]
            sent_vec = func(sent_hiddens)  # [hidden]

            hiddens.append(sent_vec)
            global_states["hidden"] = []
        results = torch.stack(hiddens, dim=0).cpu().numpy()
    return results  # [batch, hidden]


# SentEval prepare and batcher
def prepare(params, samples):

    logging.info("loading BERT model %s", params.args.model)
    params.tokenizer = BertTokenizer.from_pretrained(
        params.args.TOKENIZER_PATH_TEMPLATE.format(params.args.model),
        do_lower_case="uncased" in params.args.model,
    )
    params.bert = BertModel.from_pretrained(
        params.args.MODEL_PATH_TEMPLATE.format(params.args.model)
    )
    device = "cuda" if params.args.cuda else "cpu"
    params.bert.to(device)
    # params.
    # torch.set_grad_enabled(False)
    params.bert.eval()

    global_states["attention"] = None
    global_states["hidden"] = []

    # params.cache = CacheList()

    # for example in tqdm(samples, ascii=True, dynamic_ncols=True):
    #     example = " ".join(example)
    #     if example in params.cache.cache_list:
    #         continue

    #     tokens = params.tokenizer.tokenize(example)
    #     tokens = ["[CLS]"] + tokens + ["[SEP]"]
    #     ids = params.tokenizer.convert_tokens_to_ids(tokens)
    #     types = [0] * len(ids)
    #     mask = [1] * len(ids)

    #     id_t = torch.tensor([ids], dtype=torch.long)
    #     mask_t = torch.tensor([mask], dtype=torch.long)
    #     type_t = torch.tensor([types], dtype=torch.long)

    #     _ = params.bert(id_t, token_type_ids=type_t, attention_mask=mask_t)

    #     params.cache.add(
    #         example, global_states["hidden"]
    #     )  # list of [1, len, hidden]
    #     global_states["hidden"] = []

    return


def batcher(params, batch):

    # cache = params.cache

    hidden_func = get_hidden_batch if params.args.batch else get_hidden
    # return

    # embeddings = []

    # for tokens in batch:
    #     example = " ".join(tokens)
    #     if example in params.cache.cache_list:
    #         data = cache.get(example)
    #         hidden_t = data[BERT_LAYER]
    #         embeddings.append(hidden_t[0].mean(0))
    #     else:
    #         raise RuntimeError("Unseen example {}".format(example))

    # embeddings = torch.stack(embeddings, dim=0)

    mean_func = lambda x: torch.mean(x, dim=0)
    max_func = lambda x: torch.max(x, dim=0)[0]
    first_func = lambda x: x[0].clone()
    last_func = lambda x: x[-1].clone()

    funcs = {
        "mean": mean_func,
        "first": first_func,
        "max": max_func,
        "last": last_func,
    }
    return hidden_func(params, batch, funcs[params.args.method])


def format(results):
    # header: STS12 p, STS12 s, STS13 p, STS13 s, STS14 p, STS14 s, STS15 p, STS15 s, STS 16 p, STS16 s, STS B p, STS B s, STS B m, SICK-R p, SICK-R s, SICK-P m

    headers = "STS12 p, STS12 s, STS13 p, STS13 s, STS14 p, STS14 s, STS15 p, STS15 s, STS 16 p, STS16 s, STS B p, STS B s, STS B m, SICK-R p, SICK-R s, SICK-P m"
    values = []
    if "STS12" in results:
        values.append(
            "{:.4f}".format(results["STS12"]["all"]["pearson"]["wmean"])
        )
        values.append(
            "{:.4f}".format(results["STS12"]["all"]["spearman"]["wmean"])
        )
    else:
        values.append("na")
        values.append("na")

    if "STS13" in results:
        values.append(
            "{:.4f}".format(results["STS13"]["all"]["pearson"]["wmean"])
        )
        values.append(
            "{:.4f}".format(results["STS13"]["all"]["spearman"]["wmean"])
        )
    else:
        values.append("na")
        values.append("na")

    if "STS14" in results:
        values.append(
            "{:.4f}".format(results["STS14"]["all"]["pearson"]["wmean"])
        )
        values.append(
            "{:.4f}".format(results["STS14"]["all"]["spearman"]["wmean"])
        )
    else:
        values.append("na")
        values.append("na")

    if "STS15" in results:
        values.append(
            "{:.4f}".format(results["STS15"]["all"]["pearson"]["wmean"])
        )
        values.append(
            "{:.4f}".format(results["STS15"]["all"]["spearman"]["wmean"])
        )
    else:
        values.append("na")
        values.append("na")

    if "STS16" in results:
        values.append(
            "{:.4f}".format(results["STS16"]["all"]["pearson"]["wmean"])
        )
        values.append(
            "{:.4f}".format(results["STS16"]["all"]["spearman"]["wmean"])
        )
    else:
        values.append("na")
        values.append("na")

    if "STSBenchmark" in results:
        values.append("{:.4f}".format(results["STSBenchmark"]["pearson"]))
        values.append("{:.4f}".format(results["STSBenchmark"]["spearman"]))
        values.append("{:.4f}".format(results["STSBenchmark"]["mse"]))
    else:
        values.append("na")
        values.append("na")
        values.append("na")

    if "SICKRelatedness" in results:
        values.append("{:.4f}".format(results["SICKRelatedness"]["pearson"]))
        values.append("{:.4f}".format(results["SICKRelatedness"]["spearman"]))
        values.append("{:.4f}".format(results["SICKRelatedness"]["mse"]))
    else:
        values.append("na")
        values.append("na")
        values.append("na")

    logging.info(
        ",".join(
            ["{}={}".format(h, v) for h, v in zip(headers.split(","), values)]
        )
    )
    logging.info(headers)
    logging.info(",".join(values))

    # header: MR, CR, SUBJ, MPQA, SST-B, SST-F, TREC, SICK-E, SNLI, MRPC, MRPC f
    headers = (
        "MR, CR, SUBJ, MPQA, SST-B, SST-F, TREC, SICK-E, SNLI, MRPC, MRPC f"
    )
    values = []
    if "MR" in results:
        values.append("{:.2f}".format(results["MR"]["acc"]))
    else:
        values.append("na")

    if "CR" in results:
        values.append("{:.2f}".format(results["CR"]["acc"]))
    else:
        values.append("na")

    if "SUBJ" in results:
        values.append("{:.2f}".format(results["SUBJ"]["acc"]))
    else:
        values.append("na")

    if "MPQA" in results:
        values.append("{:.2f}".format(results["MPQA"]["acc"]))
    else:
        values.append("na")

    if "SST2" in results:
        values.append("{:.2f}".format(results["SST2"]["acc"]))
    else:
        values.append("na")

    if "SST5" in results:
        values.append("{:.2f}".format(results["SST5"]["acc"]))
    else:
        values.append("na")

    if "TREC" in results:
        values.append("{:.2f}".format(results["TREC"]["acc"]))
    else:
        values.append("na")

    if "SICKEntailment" in results:
        values.append("{:.2f}".format(results["SICKEntailment"]["acc"]))
    else:
        values.append("na")

    if "SNLI" in results:
        values.append("{:.2f}".format(results["SNLI"]["acc"]))
    else:
        values.append("na")

    if "MRPC" in results:
        values.append("{:.2f}".format(results["MRPC"]["acc"]))
        values.append("{:.2f}".format(results["MRPC"]["f1"]))
    else:
        values.append("na")
        values.append("na")

    logging.info(
        ",".join(
            ["{}={}".format(h, v) for h, v in zip(headers.split(","), values)]
        )
    )
    logging.info(headers)
    logging.info(",".join(values))

    # header: COCO r1i2t, COCO r5i2t, COCO r10i2t, COCO medr_i2t, COCO r1t2i, COCO r5t2i, COCO r10t2i, COCO medr_t2i
    headers = "COCO r1i2t, COCO r5i2t, COCO r10i2t, COCO medr_i2t, COCO r1t2i, COCO r5t2i, COCO r10t2i, COCO medr_t2i"
    if "ImageCaptionRetrieval" in results:
        values = list(results["ImageCaptionRetrieval"]["acc"][0]) + list(
            results["ImageCaptionRetrieval"]["acc"][1]
        )
        values = ["{:.2f}".format(v) for v in values]
    else:
        values = ["na"] * 10

    logging.info(
        ",".join(
            ["{}={}".format(h, v) for h, v in zip(headers.split(","), values)]
        )
    )
    logging.info(headers)
    logging.info(",".join(values))

    # header: SentLen, WC, TreeDepth, TopConst, BShift, Tense, SubjNum, ObjNum, SOMO, CoordInv, average

    headers = "SentLen, WC, TreeDepth, TopConst, BShift, Tense, SubjNum, ObjNum, SOMO, CoordInv, average"
    values = []

    for task in [
        "Length",
        "WordContent",
        "Depth",
        "TopConstituents",
        "BigramShift",
        "Tense",
        "SubjNumber",
        "ObjNumber",
        "OddManOut",
        "CoordinationInversion",
    ]:
        if task in results:
            values.append(results[task]["acc"])
        else:
            values.append("na")

    if "na" in values:
        values.append("na")
    else:
        values.append(sum(values) / len(values))

    values = ["{:.2f}".format(v) if v != "na" else v for v in values]

    logging.info(
        ",".join(
            ["{}={}".format(h, v) for h, v in zip(headers.split(","), values)]
        )
    )
    logging.info(headers)
    logging.info(",".join(values))


def main():
    args = get_args()

    # Set params for SentEval
    params_senteval = {
        "task_path": PATH_TO_DATA,
        "usepytorch": args.cuda,
        "kfold": 5,
    }
    params_senteval["classifier"] = {
        "nhid": 0,
        "optim": "rmsprop",
        "batch_size": 128,
        "tenacity": 3,
        "epoch_size": 2,
    }

    params_senteval["args"] = args

    fh = logging.FileHandler(f"log-{args.method}-{args.model}.txt")
    ch = logging.StreamHandler()

    # Set up logger
    logging.basicConfig(
        format="%(asctime)s : %(message)s",
        level=logging.DEBUG,
        handlers=[fh, ch],
    )

    # for model in BERT_MODELS:
    # BERT_MODEL = model
    for layer in range(13):
        args.layer = layer
        # BERT_LAYER = layer

        logging.info("*" * 80)
        logging.info("*" * 80)
        logging.info("*" * 80)
        logging.info("layer %d", layer)
        logging.info("*" * 80)
        logging.info("*" * 80)
        logging.info("*" * 80)

        se = senteval.engine.SE(params_senteval, batcher, prepare)
        transfer_tasks = [
            "STS12",
            "STS13",
            "STS14",
            "STS15",
            "STS16",
            "MR",
            "CR",
            "MPQA",
            "SUBJ",
            "SST2",
            "SST5",
            "TREC",
            "MRPC",
            "SICKEntailment",
            "SICKRelatedness",
            "STSBenchmark",
            "SNLI",
            "ImageCaptionRetrieval",
        ]
        probe_tasks = [
            "Length",
            "WordContent",
            "Depth",
            "TopConstituents",
            "BigramShift",
            "Tense",
            "SubjNumber",
            "ObjNumber",
            "OddManOut",
            "CoordinationInversion",
        ]
        # results = se.eval(["ImageCaptionRetrieval"])
        results = se.eval(transfer_tasks + probe_tasks)
        print(results)
        logging.info("total results: %s", str(results).replace("\n", " "))
        format(results)


if __name__ == "__main__":
    main()
