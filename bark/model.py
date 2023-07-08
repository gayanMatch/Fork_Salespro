"""
Much of this code is adapted from Andrej Karpathy's NanoGPT
(https://github.com/karpathy/nanoGPT)
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

SEMANTIC_VOCAB_SIZE = 10_000


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x, past_kv=None, use_cache=False):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        if past_kv is not None:
            past_key = past_kv[0]
            past_value = past_kv[1]
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        FULL_T = k.shape[-2]

        if use_cache is True:
            present = (k, v)
        else:
            present = None

        if past_kv is not None:
            # When `past_kv` is provided, we're doing incremental decoding and `q.shape[2] == 1`: q only contains
            # the query for the last token. scaled_dot_product_attention interprets this as the first token in the
            # sequence, so if is_causal=True it will mask out all attention from it. This is not what we want, so 
            # to work around this we set is_causal=False.
            is_causal = False
        else:
            is_causal = True

        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout, is_causal=is_causal)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return (y, present)


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.layer_idx = layer_idx

    def forward(self, x, past_kv=None, use_cache=False):
        attn_output, prev_kvs = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return (x, prev_kvs)


@dataclass
class GPTConfig:
    # config
    block_size: int = 1024
    input_vocab_size: int = 10_048
    output_vocab_size: int = 10_048
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class TransformerBlocks(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([Block(config, idx) for idx in range(config.n_layer)])

    def forward(self, x, past_kv):
        new_kv = ()
        for i, (block, past_layer_kv) in enumerate(zip(self.layers, past_kv)):
            x, kv = block(x, past_kv=past_layer_kv, use_cache=True)
            new_kv = new_kv + (kv,)
        return x, new_kv


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.input_vocab_size is not None
        assert config.output_vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.input_vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            # h=nn.ModuleList([Block(config, idx) for idx in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.transformer_h = TransformerBlocks(config)
        self.transformer_h_traced = None
        self.lm_head = nn.Linear(config.n_embd, config.output_vocab_size, bias=False)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(self, idx, merge_context=False, past_kv=None, position_ids=None, use_cache=False):
        device = idx.device
        b, t = idx.size()
        if past_kv is not None:
            assert t == 1
            tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        else:
            if merge_context:
                assert (idx.shape[1] >= 256 + 256 + 1)
                t = idx.shape[1] - 256
            else:
                assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

            # forward the GPT model itself
            if merge_context:
                tok_emb = torch.cat([
                    self.transformer.wte(idx[:, :256]) + self.transformer.wte(idx[:, 256:256 + 256]),
                    self.transformer.wte(idx[:, 256 + 256:])
                ], dim=1)
            else:
                tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        if past_kv is None:
            past_length = 0
            past_kv = tuple([None] * len(self.transformer_h.layers))
        else:
            past_length = past_kv[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(past_length, t + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)  # shape (1, t)
            assert position_ids.shape == (1, t)

        pos_emb = self.transformer.wpe(position_ids)  # position embeddings of shape (1, t, n_embd)

        x = self.transformer.drop(tok_emb + pos_emb)

        # new_kv = () if use_cache else None

        if past_kv[0] is None or self.transformer_h_traced is None:
            x, new_kv = self.transformer_h(x, past_kv)
        else:
            x, new_kv = self.transformer_h_traced(x, past_kv)

        x = self.transformer.ln_f(x)

        # inference-time mini-optimization: only forward the lm_head on the very last position
        logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim

        return (logits, new_kv)


class GPT_COARSE(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.input_vocab_size is not None
        assert config.output_vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.input_vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            # h=nn.ModuleList([Block(config, idx) for idx in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.output_vocab_size - SEMANTIC_VOCAB_SIZE, bias=False)
        self.transformer_h = TransformerBlocks(config)
        self.transformer_h_traced = None
        # self.lm_head = nn.Linear(config.n_embd, config.output_vocab_size, bias=False)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(self, idx, merge_context=False, past_kv=None, position_ids=None, use_cache=False):
        device = idx.device
        b, t = idx.size()
        if past_kv is not None:  # None
            assert t == 1
            tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
            past_length = past_kv[0][0].size(-2)
        else:
            if merge_context:  # True
                assert (idx.shape[1] >= 256 + 256 + 1)
                t = idx.shape[1] - 256
                tok_emb = torch.cat([
                    self.transformer.wte(idx[:, :256]) + self.transformer.wte(idx[:, 256:256 + 256]),
                    self.transformer.wte(idx[:, 256 + 256:])
                ], dim=1)
            else:
                assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
                tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
            past_length = 0
            past_kv = tuple([None] * len(self.transformer_h.layers))

        if position_ids is None:  # None
            position_ids = torch.arange(past_length, t + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)  # shape (1, t)
            assert position_ids.shape == (1, t)
        else:
            print("############################### OH NO ##########################################################")

        pos_emb = self.transformer.wpe(position_ids)  # position embeddings of shape (1, t, n_embd)

        x = self.transformer.drop(tok_emb + pos_emb)
        # new_kv = ()
        # x, new_kv = self.transformer_h(x, past_kv)
        # if past_kv[0] is not None:
        #     torch.jit.save(torch.jit.trace(self.transformer_h, example_inputs=(x, past_kv)), 'coarse_transformer_h')
        #     raise "Finished"
        if past_kv[0] is None or self.transformer_h_traced is None:
            x, new_kv = self.transformer_h(x, past_kv)
        else:
            x, new_kv = self.transformer_h_traced(x, past_kv)
        x = self.transformer.ln_f(x)

        # inference-time mini-optimization: only forward the lm_head on the very last position
        logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
        return logits, new_kv
