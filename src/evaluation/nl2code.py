import fire
import torch
import time
import numpy as np
import torch.distributed as dist
import multiprocessing
import concurrent.futures
import logging

from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node

from src.evaluation.nl2code_models import (
    LMModel,
    CodeSageModel, 
    TextEmbedding3Model,
)
from src.evaluation.nl2code_dataset import VectorBase
from src.utils.data_utils import CSNDataCollator, CoSQADataCollator, AdvTestDataCollator

def embedding_model_wrapper(model_name_or_path:str, args: dict):
    base_name = model_name_or_path.split("/")[-1]
    if base_name.startswith("OASIS"):
        return LMModel(model_name_or_path, pooling_strategy=args["pooling_strategy"], pooling_layer=args["pooling_layer"])
    elif base_name.startswith("codesage"):
        return CodeSageModel(model_name_or_path)
    else:
        raise NotImplementedError

@torch.no_grad()
def evaluate(
    rank: int,
    world_size: int,
    args: dict,
):
    # Init Model
    model = embedding_model_wrapper(args["model_name_or_path"], args)
    print(f"Device is {rank}.")
    model.to(rank)

    # Load query and passage data
    if "CodeSearchNet" in args["test_data_file"]:
        collate_fn = CSNDataCollator(remain_format=args["remain_format"])
    elif "CoSQA" in args["test_data_file"]:
        collate_fn = CoSQADataCollator(remain_format=args["remain_format"])
    elif "AdvTest" in args["test_data_file"]:
        collate_fn = AdvTestDataCollator(remain_format=args["remain_format"])
    else:
        raise NotImplementedError

    query_dataset = load_dataset("json", data_files=args["test_data_file"], split="train")
    query_dataset = split_dataset_by_node(query_dataset, rank=rank, world_size=world_size)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(
        query_dataset,
        sampler=query_sampler,
        collate_fn=collate_fn,
        batch_size=args["eval_batch_size"],
    )

    passage_dataset = load_dataset("json", data_files=args["candidate_database_file"], split="train")
    passage_dataset = split_dataset_by_node(passage_dataset, rank=rank, world_size=world_size)
    passage_sampler = SequentialSampler(passage_dataset)
    passage_dataloader = DataLoader(
        passage_dataset,
        sampler=passage_sampler,
        collate_fn=collate_fn,
        batch_size=args["eval_batch_size"],
    )

    query_vecs = []
    query_urls = []
    
    passage_vecs = []
    passage_urls = []

    # Query Embeddings
    for batch in tqdm(query_dataloader, position=rank, desc=f"Rank {rank} query"):
        query_urls.extend(batch["url"])
        t1 = time.time()
        query_embeddings = model.encode_queries(batch["query"])
        t2 = time.time()
        # logging.info(f"Rank {rank} query encode time: {t2 - t1}")

        query_embeddings = query_embeddings.astype(np.float32)
        query_vecs.append(query_embeddings)
        torch.cuda.empty_cache()
    query_vecs = np.concatenate(query_vecs, axis=0)

    # Passage Embeddings
    for batch in tqdm(passage_dataloader, position=rank, desc=f"Rank {rank} passage"):
        passage_urls.extend(batch["url"])
        passage_embeddings = model(batch["passage"])
        passage_embeddings = passage_embeddings.astype(np.float32)
        passage_vecs.append(passage_embeddings)
        torch.cuda.empty_cache()
    passage_vecs = np.concatenate(passage_vecs, axis=0) 

    return {
        "query_urls": query_urls,
        "query_vecs": query_vecs,
        "passage_urls": passage_urls,
        "passage_vecs": passage_vecs
    }

def evaluate_on_openai(
    args: dict,
):
    # Load query and passage data
    if "CodeSearchNet" in args["test_data_file"]:
        collate_fn = CSNDataCollator(remain_format=args["remain_format"])
    elif "CoSQA" in args["test_data_file"]:
        collate_fn = CoSQADataCollator(remain_format=args["remain_format"])
    elif "AdvTest" in args["test_data_file"]:
        collate_fn = AdvTestDataCollator(remain_format=args["remain_format"])
    else:
        raise NotImplementedError
    
    query_dataset = load_dataset("json", data_files=args["test_data_file"], split="train")
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(
        query_dataset,
        sampler=query_sampler,
        collate_fn=collate_fn,
        batch_size=args["eval_batch_size"],
    )

    passage_dataset = load_dataset("json", data_files=args["candidate_database_file"], split="train")
    passage_sampler = SequentialSampler(passage_dataset)
    passage_dataloader = DataLoader(
        passage_dataset,
        sampler=passage_sampler,
        collate_fn=collate_fn,
        batch_size=args["eval_batch_size"], 
    )

    # query_vecs = []
    query_urls = []
    queries = []
    
    # passage_vecs = []
    passage_urls = []
    passages = []

    # Query Embeddings
    for batch in query_dataloader:
        query_urls.extend(batch["url"])
        queries.extend(batch["query"])

    # Passage Embeddings
    for batch in passage_dataloader:
        passage_urls.extend(batch["url"])
        passages.extend(batch["passage"])
    
    embedding_model = TextEmbedding3Model()
    query_vecs = np.array(embedding_model.call_openai_embeddings(queries))
    passage_vecs = np.array(embedding_model.call_openai_embeddings(passages))

    return {
        "query_urls": query_urls,
        "query_vecs": query_vecs,
        "passage_urls": passage_urls,
        "passage_vecs": passage_vecs
    }

def main(
    model_name_or_path: str = None,
    test_data_file: str = None,
    candidate_database_file: str = None,
    remain_format: bool = False,
    pooling_strategy: str = "last",
    pooling_layer: int = -1,
    eval_batch_size: int = 256,
):
    args = {"model_name_or_path": model_name_or_path,
            "test_data_file": test_data_file,
            "candidate_database_file": candidate_database_file,
            "remain_format": remain_format,
            "pooling_strategy": pooling_strategy,
            "pooling_layer": pooling_layer,
            "eval_batch_size": eval_batch_size,
    }
    logging.info(f"Running with args: {args}")

    query_dataset = load_dataset("json", data_files=test_data_file, split="train")
    passage_dataset = load_dataset("json", data_files=candidate_database_file, split="train")

    # Datainfo display
    logging.info("***** Running evaluation *****")
    logging.info(f"Num queries = {len(query_dataset)}")
    logging.info(f"Num passages = {len(passage_dataset)}")
    logging.info(f"Batch size = {eval_batch_size}")

    start_time = time.time()

    if args["model_name_or_path"] == "text-embedding-3-large":
        results = evaluate_on_openai(args)
        
        query_urls = results["query_urls"]
        query_vecs = results["query_vecs"]
        passage_urls = results["passage_urls"]
        passage_vecs = results["passage_vecs"]
    else:
        world_size = torch.cuda.device_count()
        multiprocessing.set_start_method('spawn', force=True)

        with concurrent.futures.ProcessPoolExecutor(max_workers=world_size) as executor:
            results = list(executor.map(evaluate, list(range(world_size)), [world_size] * world_size, [args] * world_size))

        query_vecs = []
        query_urls = []
        
        passage_vecs = []
        passage_urls = []

        for result in results:
            query_urls.extend(result["query_urls"])
            query_vecs.append(result["query_vecs"])
            passage_urls.extend(result["passage_urls"])
            passage_vecs.append(result["passage_vecs"])
        
        query_vecs = np.concatenate(query_vecs, axis=0)
        passage_vecs = np.concatenate(passage_vecs, axis=0)

    encode_time = time.time()
    logging.info(f"Encode time: {encode_time - start_time}s")

    vector_base = VectorBase(dimension=query_vecs.shape[1])

    # Add vectors and search
    vector_base.add(passage_vecs, passage_urls)
    similarities, indices = vector_base.search(query_vecs, 1000)

    search_time = time.time()
    logging.info(f"Search time: {search_time - encode_time:.2f}s")

    # Metric display
    mrr_at_k = vector_base.calculate_mrr_at_k(query_urls, indices, max_k=1000)
    for k in range(1, 6):
        logging.info(f"MRR@{k}: {mrr_at_k[k]}")
    logging.info(f"MRR@{1000}: {mrr_at_k[1000]}")
    logging.info(f"[{mrr_at_k[1]}, {mrr_at_k[2]}, {mrr_at_k[3]}, {mrr_at_k[4]}, {mrr_at_k[5]}, {mrr_at_k[1000]}]")

    hit_rate_at_k = vector_base.calculate_hit_rate_at_k(query_urls, indices, max_k=5)
    for k, hit_rate in hit_rate_at_k.items():
        logging.info(f"Hit rate@{k}: {hit_rate}")
    logging.info(f"[{hit_rate_at_k[1]}, {hit_rate_at_k[2]}, {hit_rate_at_k[3]}, {hit_rate_at_k[4]}, {hit_rate_at_k[5]}]")
    
    return {
        "mrr_at_1000": mrr_at_k[1000],
    }

if __name__ == '__main__':
    fire.Fire(main)