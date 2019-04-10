# coding = utf-8
# author = xy

from torch import nn
import torch
from torch.nn import functional as f


class Embedding(nn.Module):
    """
    standard embedding
    input: tensor (batch_size, seq_len)
    return: tensor (seq_len, batch_size, w2v_size)
    """
    def __init__(self, embedding):
        super(Embedding, self).__init__()

        self.vocab_size = embedding.shape[0]
        self.w2v_size = embedding.shape[1]

        self.embedding_fix = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.w2v_size,
            padding_idx=0,
            _weight=torch.Tensor(embedding)
        )

        self.embedding_v = nn.Embedding(
            num_embeddings=2,
            embedding_dim=self.w2v_size,
            padding_idx=0
        )

        self.embedding_fix.weight.requires_grad = False
        self.embedding_dim = self.embedding_fix.embedding_dim

    def forward(self, tensor):
        embedding_1 = self.embedding_fix(tensor)  # (batch_size, c_len, w2v_size)
        tensor = tensor - (self.vocab_size-2)
        tensor = f.relu(tensor)
        embedding_2 = self.embedding_v(tensor)  # (batch_size, c_len, w2v_size)

        embedding = embedding_1 + embedding_2
        embedding = embedding.transpose(0, 1)

        return embedding
