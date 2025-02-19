import argparse
import torch
import deepspeed
import os
import torch.distributed as dist
import random

from deepspeed.accelerator import get_accelerator
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, SequentialSampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset, load_from_disk
from datasets.distributed import split_dataset_by_node
from tqdm import tqdm
import wandb
import requests

from src.fine_tuning.embedding_models import LMTrainingModel

from src.utils.data_utils import RawEmbeddingTrainingDataCollator
from src.utils.utils import print_rank_0, is_rank_0

def parse_args():
    parser = argparse.ArgumentParser(description="Used for embedding model training.")

    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')

    parser.add_argument('--output_dir', type=str, default="./logs/debug", help='Output directory for the model and log.')
    parser.add_argument('--model_name_or_path', default='jinaai/jina-embeddings-v2-base-code', type=str, help='Name or path for the embedding model.')

    parser.add_argument('--query_column', type=str, default="query")
    parser.add_argument('--passage_column', type=str, default="passage")
    parser.add_argument('--label_column', type=str, default="label")

    parser.add_argument('--ibn_w', type=float, default=1.0)
    parser.add_argument('--cosine_w', type=float, default=1.0)
    parser.add_argument('--mix_data', action='store_true')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--in_batch_num', type=int, default=8)
    parser.add_argument('--pooling_strategy', type=str, default="last")
    parser.add_argument('--pooling_layer', type=int, default=-1)

    parser.add_argument('--lr', type=float, default=1e-5)

    parser.add_argument('--train_data_path', default=None, type=str, help='An input train data directory.')
    parser.add_argument('--train_batch_size', default=2, type=int, help='Batch size for training.')

    parser.add_argument('--do_eval', action='store_true')

    parser.add_argument('--epochs', default=1, type=int, help='Number of epochs for training.')
    parser.add_argument('--seed', default=3407, type=int, help='Random seed for all random module.')

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--multi_node', action='store_true')

    parser.add_argument('--checkpoint', default=None, type=str, help='Path to the checkpoint to load from.')

    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    return args

def set_seed(seed):
    torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)

def get_training_model(args, tokenizer, model):
    base_name = args.model_name_or_path.split("/")[-1]
    args.base_name = base_name
    if base_name.startswith("Qwen2") or base_name.startswith("deepseek-coder") or base_name.startswith("Llama"):
        return LMTrainingModel(tokenizer, model, args.ibn_w, args.cosine_w, args.max_length, args.in_batch_num, args.pooling_strategy, args.pooling_layer)
    else:
        raise NotImplementedError

def checkpoint_state_check(client_state, args):
    if client_state is not None:
        for key, value in client_state["args"].items():
            if key not in args.__dict__:
                raise ValueError(f"Argument {key} is not in the current argument list.")
            if args.__dict__[key] != value:
                raise ValueError(f"Argument {key} is not the same as the checkpoint argument value.")
    return

def train():
    args = parse_args()
    set_seed(args.seed)

    # Backend initialization
    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()
    
    print_rank_0(f"Arguments{args}")
    
    # Model initialization
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, output_hidden_states=True)
    model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True, output_hidden_states=True)
    model = get_training_model(args, tokenizer, model)
    model.train()

    # Dataset loading
    streaming = False
    if args.train_data_path.endswith(".json"):
        dataset = load_dataset(path=args.train_data_path, split="train", streaming=streaming)
    else:
        dataset = load_from_disk(args.train_data_path)

    # Data loader init
    sampler = DistributedSampler(dataset)
    data_collator = RawEmbeddingTrainingDataCollator(args.query_column, args.passage_column, args.label_column, "lang")
    train_dataloader = DataLoader(dataset,
                                    collate_fn=data_collator,
                                    batch_size=args.train_batch_size,
                                    shuffle=(sampler is None),
                                    sampler=sampler,
                                 )
    
    # Optimizer initialization
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2, eps=1e-3)

    # DeepSpeed initialization
    model_engine, *_ = deepspeed.initialize(args=args, model=model, optimizer=optimizer)

    checkpoint_global_step = -1
    if args.checkpoint is not None:
        _, client_state = model_engine.load_checkpoint(args.checkpoint)
        print_rank_0(client_state["args"])
        checkpoint_state_check(client_state, args)
        checkpoint_global_step = client_state["global_step"]
    print_rank_0(f"Global step {checkpoint_global_step} from checkpoint {args.checkpoint}.")
    
    model_engine.enable_input_require_grads()
    model_engine.gradient_checkpointing_enable()

    # Train!
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for step, batch in enumerate(tqdm(train_dataloader, disable=(torch.distributed.get_rank() != 0))):
            if len(train_dataloader) * epoch + step < checkpoint_global_step + 1:
                continue
            if args.mix_data:
                outputs = model_engine(batch["sequences"], batch["labels"], batch["gt"])
            else:
                outputs = model_engine(batch["sequences"])
            loss = outputs["loss"]
            model_engine.backward(loss)
            model_engine.step()
            print_rank_0(f"Epoch {epoch + 1}, Step {step}, Loss {loss.item()}")
            print_rank_0(f"Epoch {epoch + 1}, Step {step}, ibn_loss {outputs['ibn_loss'].item() / args.ibn_w if args.ibn_w != 0 else 0} \
            , cosine_loss {outputs['cosine_loss'].item() / args.cosine_w if args.cosine_w != 0 else 0}")

            if (step + 1) % 2500 == 0:
                client_state = {
                    "global_step": len(train_dataloader) * epoch + step, 
                    "args": {
                        "model_name_or_path": args.model_name_or_path,
                        "train_data_path": args.train_data_path,
                        "train_batch_size": args.train_batch_size,
                        "ibn_w": args.ibn_w,
                        "cosine_w": args.cosine_w,
                        "mix_data": args.mix_data,
                        "max_length": args.max_length,
                        "in_batch_num": args.in_batch_num,
                        "pooling_strategy": args.pooling_strategy,
                        "pooling_layer": args.pooling_layer,
                        "lr": args.lr,
                        "seed": args.seed,
                    }
                }
                model_engine.save_checkpoint(os.path.join(args.output_dir, "checkpoint"), client_state=client_state)
                if is_rank_0():
                    os.makedirs(os.path.join(args.output_dir, f"{args.base_name}_epoch_{epoch + 1}_step{step + 1}"), exist_ok=True)
                    model_engine.base_model.save_pretrained(os.path.join(args.output_dir, f"{args.base_name}_epoch_{epoch + 1}_step{step + 1}"))
                    model_engine.tokenizer.save_pretrained(os.path.join(args.output_dir, f"{args.base_name}_epoch_{epoch + 1}_step{step + 1}"))

        if is_rank_0():
            os.makedirs(os.path.join(args.output_dir, f"{args.base_name}_epoch_{epoch + 1}"), exist_ok=True)
            model_engine.base_model.save_pretrained(os.path.join(args.output_dir, f"{args.base_name}_epoch_{epoch + 1}"))
            model_engine.tokenizer.save_pretrained(os.path.join(args.output_dir, f"{args.base_name}_epoch_{epoch + 1}"))

    torch.cuda.empty_cache()
    dist.barrier()

if __name__ == "__main__":
    train()