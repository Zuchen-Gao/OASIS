import torch

import torch.nn.functional as F
import torch.distributed as dist
from typing import List, Optional

from torch import nn, Tensor

def cosine_loss(y_pred: torch.Tensor, y_true: torch.Tensor, tau: float = 20.0) -> torch.Tensor:
    """
    Compute cosine loss

    :param y_true: torch.Tensor, ground truth.
        The y_true must be zigzag style, such as [o[0][0], o[0][1], o[1][0], o[1][1], ...], where (o[0][0], o[0][1]) stands for a pair.
    :param y_pred: torch.Tensor, model output.
        The y_pred must be zigzag style, such as [l[0][0], l[0][1], l[1][0], l[1][1], ...], where (l[0][0], l[0][1]) stands for a pair.
    :param tau: float, scale factor, default 20

    :return: torch.Tensor, loss value
    """  # NOQA
    # modified from: https://github.com/bojone/CoSENT/blob/124c368efc8a4b179469be99cb6e62e1f2949d39/cosent.py#L79
    y_true = y_true[::2, 0]
    y_true = (y_true[:, None] < y_true[None, :]).float()
    y_pred = F.normalize(y_pred, p=2, dim=1)
    y_pred = torch.sum(y_pred[::2] * y_pred[1::2], dim=1) * tau
    y_pred = y_pred[:, None] - y_pred[None, :]
    y_pred = (y_pred - (1 - y_true) * 1e12).view(-1)
    zero = torch.Tensor([0]).to(y_pred.device)
    y_pred = torch.concat((zero, y_pred), dim=0)
    return torch.logsumexp(y_pred, dim=0)

def bimodal_in_batch_negative_loss(y_pred: torch.Tensor,
                                   tau: float = 20) -> torch.Tensor:
    """
    Compute cosine loss

    :param y_pred: torch.Tensor, model output.
        The y_pred must be zigzag style, such as [o[0][0], o[0][1], o[1][0], o[1][1], ...], where (o[0][0], o[0][1]) stands for a pair.
    :param tau: float, scale factor, default 20

    :return: torch.Tensor, loss value
    """
    device = y_pred.device

    y_pred = F.normalize(y_pred, dim=1, p=2)
    similarities = y_pred @ y_pred.T
    similarities = similarities * tau
    exp_similarities = torch.exp(similarities)

    idxs = torch.arange(0, y_pred.shape[0]).int().to(device)
    idxs_1 = idxs[None, :]
    idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
    pos_mask = (idxs_1 == idxs_2).float()
    neg_mask = 1 - (pos_mask + torch.eye(y_pred.shape[0]).to(device))

    gamma_weight = F.normalize(exp_similarities * neg_mask, dim=1, p=1)
    cs_similarities = exp_similarities * (gamma_weight + pos_mask)
    cs_softmax = F.normalize(cs_similarities, dim=1, p=1) * pos_mask

    loss = -torch.log(cs_softmax.sum(1)).mean()

    return loss

def bimodal_in_batch_negative_loss_with_label(y_pred: torch.Tensor,
                                            y_true: torch.Tensor,
                                            tau: float = 20) -> torch.Tensor:
    """
    Compute cosine loss

    :param y_pred: torch.Tensor, model output.
        The y_pred must be zigzag style, such as [o[0][0], o[0][1], o[1][0], o[1][1], ...], where (o[0][0], o[0][1]) stands for a pair.
    :param y_true: torch.Tensor, label for output.
        The y_true must be zigzag style, such as [l[0][0], l[0][1], l[1][0], l[1][1], ...], where (l[0][0], l[0][1]) are the same.
    :param tau: float, scale factor, default 20

    :return: torch.Tensor, loss value
    """
    device = y_pred.device
    y_true = (y_true == 1).int()

    y_pred = F.normalize(y_pred, dim=1, p=2)
    similarities = y_pred @ y_pred.T
    similarities = similarities * tau
    exp_similarities = torch.exp(similarities)

    idxs = torch.arange(0, y_pred.shape[0]).int().to(device)
    idxs_1 = idxs[None, :]
    idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
    idxs_1 *= y_true.T
    idxs_1 += (y_true.T == 0).int() * -2
    idxs_2 *= y_true
    idxs_2 += (y_true == 0).int() * -1
    pos_mask = (idxs_1 == idxs_2).float()
    neg_mask = 1 - (pos_mask + torch.eye(y_pred.shape[0]).to(device))

    gamma_weight = F.normalize(exp_similarities * neg_mask, dim=1, p=1)
    cs_similarities = exp_similarities * (gamma_weight + pos_mask)
    cs_softmax = F.normalize(cs_similarities, dim=1, p=1) * pos_mask

    cs_softmax_value = cs_softmax.sum(1)
    cs_softmax_valid = torch.where(cs_softmax_value == 0, torch.ones_like(cs_softmax_value), cs_softmax_value)

    loss = -torch.log(cs_softmax_valid).sum() / torch.sum(cs_softmax_value != 0)

    return loss

class EmbeddingModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.config = base_model.config
        self.base_model = base_model
    
    def gradient_checkpointing_enable(self):
        self.base_model.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        self.base_model.gradient_checkpointing_disable()
    
    def enable_input_require_grads(self):
        self.base_model.enable_input_require_grads()
    
    def compute_loss(self,
                     outputs: torch.Tensor,
                     labels: torch.Tensor
                     ):
        raise NotImplementedError
    
    def forward(self, **inputs):
        raise NotImplementedError

class LMTrainingModel(EmbeddingModel):
    def __init__(self, tokenizer, base_model, ibn_w=1, cosine_w=1, max_length=512, in_batch_num=8, pooling_strategy="last", pooling_layer=-1):
        super().__init__(base_model)
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.ibn_w = ibn_w
        self.cosine_w = cosine_w

        self.in_batch_num = in_batch_num
        self.pooling_strategy = pooling_strategy
        self.pooling_layer = pooling_layer

        self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def compute_loss(self,
                     outputs: torch.Tensor,
                     labels: Optional[torch.Tensor] = None,
                     gt_labels: Optional[torch.Tensor] = None,
                     ):
        if labels is not None:
            ibn_loss = torch.tensor(0.)
            cos_loss = torch.tensor(0.)
            if self.ibn_w > 0:
                if gt_labels is not None:
                    ibn_loss = self.ibn_w * bimodal_in_batch_negative_loss_with_label(outputs, gt_labels)
                else:
                    ibn_loss = self.ibn_w * bimodal_in_batch_negative_loss_with_label(outputs, labels)
            if self.cosine_w > 0:
                cos_loss = self.cosine_w * cosine_loss(outputs, labels)
            return ibn_loss + cos_loss, ibn_loss, cos_loss
        else:
            ibn_loss = bimodal_in_batch_negative_loss(outputs)
            return ibn_loss, ibn_loss, torch.tensor(0.)
        
    
    def get_detailed_instruct(self, query: str) -> str:
        task_description = 'Given a code search query, retrieve relevant code snippet that answer the query'
        return f'Instruct: {task_description}\nQuery: {query}'
    
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

    def forward(self, sequences: List[str], labels: Optional[List[float]] = None, gt_labels: Optional[List[bool]] = None):
        for query_i in range(0, int(len(sequences)), 2):
            sequences[query_i] = self.get_detailed_instruct(sequences[query_i])
        inputs = self.tokenizer(sequences, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.base_model.device) for k, v in inputs.items()}
        outputs = self.base_model(**inputs, return_dict=True)

        attention_mask = inputs["attention_mask"]
        if self.pooling_strategy == "last":
            embeddings = self.last_token_pool(outputs.hidden_states[self.pooling_layer], attention_mask)
        elif self.pooling_strategy == "mean":
            embeddings = self.mean_pool(outputs.hidden_states[self.pooling_layer], attention_mask)
        elif self.pooling_strategy == "first":
            embeddings = self.first_token_pool(outputs.hidden_states[self.pooling_layer], attention_mask)
        else:
            raise NotImplementedError

        # Embedding gather
        dist.barrier()

        embeddings_list = [torch.ones_like(embeddings) for _ in range(dist.get_world_size())]
        dist.all_gather(embeddings_list, embeddings.contiguous())

        embeddings_list[dist.get_rank()] = embeddings

        # random in-batch samples
        node_slice = [i % dist.get_world_size() for i in range(dist.get_rank(), dist.get_rank() + self.in_batch_num)]
        embeddings_list = [embeddings_list[i] for i in node_slice]

        embeddings_gather = torch.cat(embeddings_list)

        # Label gather
        if labels is not None:
            labels = torch.tensor(labels).unsqueeze(-1).to(self.base_model.device)

            labels_list = [torch.ones_like(labels) for _ in range(dist.get_world_size())]
            dist.all_gather(labels_list, labels.contiguous())

            # random in-batch samples
            labels_list = [labels_list[i] for i in node_slice]

            labels_gather = torch.cat(labels_list)
        else:
            labels_gather = None
        
        # gt_label gather
        if gt_labels is not None:
            gt_labels = torch.tensor(gt_labels).unsqueeze(-1).to(self.base_model.device)

            gt_labels_list = [torch.ones_like(gt_labels) for _ in range(dist.get_world_size())]
            dist.all_gather(gt_labels_list, gt_labels.contiguous())

            # random in-batch samples
            gt_labels_list = [gt_labels_list[i] for i in node_slice]

            gt_labels_gather = torch.cat(gt_labels_list)
            gt_labels_gather = gt_labels_gather.int()
        else:
            gt_labels_gather = None

        # Compute loss
        loss, ibn_loss, cosine_loss = self.compute_loss(embeddings_gather, labels_gather, gt_labels_gather)
        return {
            "loss": loss,
            "ibn_loss": ibn_loss,
            "cosine_loss": cosine_loss,
            "outputs": embeddings
        }