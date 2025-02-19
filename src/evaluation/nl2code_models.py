import torch
import concurrent.futures
import logging
import tiktoken
import time

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

from openai import APIConnectionError, OpenAI, AzureOpenAI
from angle_emb import AnglE

class TextEmbedding3Model():
    def __init__(self, use_azure=False):
        self.use_azure = use_azure

    azure_endpoint_key_model = [
        ["endpoint", "key", "text-embedding-3-large"],
        ["endpoint", "key", "text-embedding-3-large"],
    ]
    openai_api_key_url_model = ["api_key", "url", "text-embedding-3-large"]

    def truncate_text(self, text, max_length=8192):
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        return encoding.decode(tokens[:max_length])
    
    def generate_embeddings_response(self, texts):
        client = OpenAI(
            api_key=self.openai_api_key_url_model[0],
            base_url=self.openai_api_key_url_model[1],
            timeout=60,
        )

        response = client.embeddings.create(
            input=texts, 
            model=openai_api_key_url_model[2]
        )
    return response

    def generate_azure_embeddings_response(self, texts, index=0):
        client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint_key_model[index][0],
            api_key=self.azure_endpoint_key_model[index][1],
            api_version="2024-02-01",
            timeout=60,
        )

        response = client.embeddings.create(
            input=texts, 
            model=self.azure_endpoint_key_model[index][2]
        )
        return response

    def send_embeddings_request(self, texts, index=0):
        attempt = 0
        max_attempts = 5
        while attempt < max_attempts:
            try:
                index = index % len(self.azure_endpoint_key_model)
                if self.use_azure:
                    response_message = self.generate_azure_embeddings_response(texts, index)
                else:
                    response_message = self.generate_embeddings_response(texts)
                if response_message is None:
                    attempt += 1
                    continue
                embeddings = []
                for i in range(len(texts)):
                    embedding = response_message.data[i].embedding
                    if isinstance(embedding, list) and isinstance(embedding[0], float):
                        embeddings.append(embedding)
                    else:
                        embeddings.append(None)
                return embeddings # [[int,...],...]

            except APIConnectionError as e:
                logging.error(f"Connection error: {e}. Attempt {attempt + 1} of {max_attempts}")
                # Retry after 7 seconds
                time.sleep(7)
                attempt += 1
                if attempt == max_attempts:
                    return [None] * len(texts)
                else:
                    continue  # Try to request again

            except Exception as e:
                logging.error(f"An unknown error occurred: {e}. \nAttempt {attempt + 1} of {max_attempts}")
                # Retry after 10 seconds
                time.sleep(10)
                attempt += 1
                if attempt == max_attempts:
                    return [None] * len(texts)

    def call_openai_embeddings(self, texts):
        '''
        [text,...] -> [[float,...],...]
        '''
        texts = [self.truncate_text(text) for text in texts]

        chunk_size = 256
        workers_num = 15
        chunked_texts = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers_num) as executor:
            responses = list(executor.map(self.send_embeddings_request, chunked_texts, list(range(len(chunked_texts)))))
            embeddings = []
            for batched_embeddings in responses:
                for embedding in batched_embeddings:
                    embeddings.append(embedding)
            return embeddings

class LMModel():
    def __init__(self, model_name_or_path: str, pooling_strategy="last", pooling_layer=-1):
        self.model = AutoModel.from_pretrained(model_name_or_path, output_hidden_states=True, trust_remote_code=True, torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

        self.tokenizer.padding_side = "left"
        self.pooling_strategy = pooling_strategy
        self.pooling_layer = pooling_layer
        self.max_length = 8192

    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    def first_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            sequence_lengths = attention_mask.sum(dim=1)
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), -sequence_lengths]
        else:
            return last_hidden_states[:, 0]
    
    def mean_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        embeddings = (last_hidden_states * attention_mask[:, :, None]).sum(1) / attention_mask.sum(1)[:, None]
        return embeddings

    def get_detailed_instruct(self, query: str) -> str:
        task_description = 'Given a code search query, retrieve relevant code snippet that answer the query'
        return f'Instruct: {task_description}\nQuery: {query}'
    
    @torch.no_grad()
    def __call__(self, sequences: list[str]):
        inputs = self.tokenizer(sequences, max_length=self.max_length, padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs, return_dict=True)
        if self.pooling_strategy == "last":
            embeddings = self.last_token_pool(outputs.hidden_states[self.pooling_layer], inputs['attention_mask'])
        elif self.pooling_strategy == "first":
            embeddings = self.first_token_pool(outputs.hidden_states[self.pooling_layer], inputs['attention_mask'])
        elif self.pooling_strategy == "mean":
            embeddings = self.mean_pool(outputs.hidden_states[self.pooling_layer], inputs['attention_mask'])
        else:
            raise ValueError("Invalid pooling strategy")
        return embeddings.detach().to(torch.float).cpu().numpy()
    
    def encode_queries(self, queries: list[str]):
        queries = [self.get_detailed_instruct(query) for query in queries]
        return self(queries)
    
    def to(self, device):
        self.device = device
        self.model.to(device)

class CodeSageModel():
    def __init__(self, model_name_or_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True) #, torch_dtype=torch.bfloat16)
        self.device = torch.device("cpu")
    
    def _convert2ids(self, text, max_length):
        tokens = self.tokenizer.tokenize(text)[:max_length - 1]
        tokens = tokens + [self.tokenizer.eos_token]
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        padding_length = max_length - len(ids)
        ids += [self.tokenizer.pad_token_id] * padding_length
        return ids
    
    def __call__(self, sequences: list[str], max_length=1024):
        sequences = [self._convert2ids(seq, max_length) for seq in sequences]
        input_ids = torch.tensor(sequences).to(self.device)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return outputs.pooler_output.float().cpu().detach().numpy()

    def encode_queries(self, queries: list[str]):
        return self(queries, 128)
    
    def to(self, device):
        self.device = device
        self.model.to(device)