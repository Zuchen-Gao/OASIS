import fire
import torch
import os
import multiprocessing
import concurrent.futures
import logging
import numpy as np
import random

from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from src.evaluation.nl2code import embedding_model_wrapper
from src.utils.data_utils import CodeNetDataCollator
from src.evaluation.nl2code_models import TextEmbedding3Model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@torch.no_grad()
def evaluate(
    rank: int,
    world_size: int,
    args: dict,
    lang_type: str,
):  
    # default setting
    args["pooling_strategy"] = "last"
    args["pooling_layer"] = -1

    # Init Model
    model = embedding_model_wrapper(args["model_name_or_path"], args)
    print(f"Device is {rank}.")
    model.to(rank)

    collate_fn = CodeNetDataCollator(lang=args[lang_type])
    dataset_path = os.path.join(args["dataset_dir"], f"{args[lang_type]}_with_func.jsonl")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        collate_fn=collate_fn,
        batch_size=args["eval_batch_size"],
    )

    code_vecs = []
    code_labels = []
    code_indexs = []

    # Code Embeddings
    for batch in tqdm(dataloader, position=rank, desc=f"Rank {rank} code"):
        code_labels.extend(batch["label"])
        code_indexs.extend(batch["index"])
        code_embeddings = model(batch["func"])
        code_embeddings = code_embeddings.astype(np.float32)
        code_vecs.append(code_embeddings)
        torch.cuda.empty_cache()
    code_vecs = np.concatenate(code_vecs, axis=0)

    result = {
        "embeddings": code_vecs,
        "labels": code_labels,
        "indexs": code_indexs
    }
    return result

def evaluate_on_openai(
    args: dict,
    lang_type: str,
):  
    collate_fn = CodeNetDataCollator(lang=args[lang_type])
    dataset_path = os.path.join(args["dataset_dir"], f"{args[lang_type]}_with_func.jsonl")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        collate_fn=collate_fn,
        batch_size=args["eval_batch_size"],
    )

    codes = []
    code_labels = []
    code_indexs = []

    # Code Embeddings
    for batch in tqdm(dataloader):
        code_labels.extend(batch["label"])
        code_indexs.extend(batch["index"])
        codes.extend(batch["func"])
    
    embedding_model = TextEmbedding3Model()
    code_vecs = np.array(embedding_model.call_openai_embeddings(codes))

    result = {
        "embeddings": code_vecs,
        "labels": code_labels,
        "indexs": code_indexs
    }
    return result


def get_map_score(query_vecs, query_labels, query_indexs, candidate_vecs, candidate_labels, candidate_indexs):
    # Normalize vecs
    query_vecs = query_vecs / np.linalg.norm(query_vecs, axis=1, keepdims=True)
    candidate_vecs = candidate_vecs / np.linalg.norm(candidate_vecs, axis=1, keepdims=True)

    # Calculate MAP score
    scores = np.matmul(query_vecs, candidate_vecs.T)  # num_queries x num_candidates
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]

    MAP = []
    for i in range(scores.shape[0]):
        cont = 0
        label = int(query_labels[i])
        query_index = query_indexs[i]

        Avep = []
        for j, index in enumerate(list(sort_ids[i])):
            if query_index == candidate_indexs[index]:
                cont += 1
                continue
            if int(candidate_labels[index]) == label:
                Avep.append((len(Avep) + 1) / (j + 1 - cont))
        if len(Avep) != 0:
            MAP.append(sum(Avep) / len(Avep))

    map_score = float(np.mean(MAP))
    return map_score

def main(
    model_name_or_path: str = None,
    dataset_dir: str = None,
    eval_batch_size: int = 16,
    src_lang: str = "python",
    tgt_lang: str = "python",
):
    logging.info(f"Arguments: {locals()}")
    args = {
        "model_name_or_path": model_name_or_path,
        "dataset_dir": dataset_dir,
        "eval_batch_size": eval_batch_size,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
    }
    if args["model_name_or_path"] == "text-embedding-3-large":
        src_embd_dict = evaluate_on_openai(args, "src_lang")
        if src_lang == tgt_lang:
            tgt_embd_dict = src_embd_dict
        else:
            tgt_embd_dict = evaluate_on_openai(args, "tgt_lang")
    else: # regular model
        # multiprocess realrun
        world_size = torch.cuda.device_count()
        multiprocessing.set_start_method('spawn', force=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=world_size) as executor:
            src_results = list(executor.map(
                evaluate, 
                list(range(world_size)), 
                [world_size] * world_size, 
                [args] * world_size, 
                ["src_lang"] * world_size))
            if src_lang == tgt_lang:
                tgt_results = src_results
            else:
                tgt_results = list(executor.map(
                    evaluate, 
                    list(range(world_size)), 
                    [world_size] * world_size, 
                    [args] * world_size, 
                    ["tgt_lang"] * world_size))

        src_embd_dict = { "embeddings": [], "labels": [], "indexs": []}
        tgt_embd_dict = { "embeddings": [], "labels": [], "indexs": []}

        for src_result in src_results:
            src_embd_dict["embeddings"].append(src_result["embeddings"])
            src_embd_dict["labels"].extend(src_result["labels"])
            src_embd_dict["indexs"].extend(src_result["indexs"])
        src_embd_dict["embeddings"] = np.concatenate(src_embd_dict["embeddings"], axis=0)

        for tgt_result in tgt_results:
            tgt_embd_dict["embeddings"].append(tgt_result["embeddings"])
            tgt_embd_dict["labels"].extend(tgt_result["labels"])
            tgt_embd_dict["indexs"].extend(tgt_result["indexs"])
        tgt_embd_dict["embeddings"] = np.concatenate(tgt_embd_dict["embeddings"], axis=0)

        # single process debug
        # src_embd_dict = evaluate(0, 1, args, "src_lang")
        # if src_lang == tgt_lang:
        #     tgt_embd_dict = src_embd_dict
        # else:
        #     tgt_embd_dict = evaluate(0, 1, args, "tgt_lang")
    
    # Calculate MAP score
    eval_map_score = get_map_score(
        query_vecs=src_embd_dict["embeddings"],
        query_labels=src_embd_dict["labels"],
        query_indexs=src_embd_dict["indexs"],
        candidate_vecs=tgt_embd_dict["embeddings"],
        candidate_labels=tgt_embd_dict["labels"],
        candidate_indexs=tgt_embd_dict["indexs"]
    )
    logging.info(f"Src lang:{src_lang}, Tgt lang:{tgt_lang}, MAP score: {eval_map_score}")
    return eval_map_score

if __name__ == "__main__":
    fire.Fire(main)