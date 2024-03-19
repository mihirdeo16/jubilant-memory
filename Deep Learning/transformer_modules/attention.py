#!/usr/bin/env python3
"""
We are going to implement the attention mechanism in the transformer architecture
"""

import torch
import torch.nn as nn
__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"

class SelfAttention(nn.Module):
    """
    The key/value/query concepts come from retrieval systems. 
    For example, when you type a query to search for some video on Youtube, the search engine will map your query against a set of keys (video title, description etc.) associated with candidate videos in the database, then present you the best matched videos (values).
    """

    def __init__(self, input_dim, dropout_value=0.1) -> None:

        super(SelfAttention, self).__init__()

        self.input_dim = input_dim

        self.query_layer = nn.Linear(input_dim, input_dim)
        self.key_layer = nn.Linear(input_dim, input_dim)
        self.value_layer = nn.Linear(input_dim, input_dim)

        self.dropout = nn.Dropout(dropout_value)

    def scaled_dot_product_attention(self, query, key, value, mask):

        atten_scores = torch.matmul(query, key.transpose(-2, -1))

        atten_scores_scaled = torch.divide(
            atten_scores, torch.sqrt(torch.tensor(self.input_dim)))

        if mask is not None:
            atten_scores_scaled = atten_scores_scaled.masked_fill(
                mask == 0, -1e9)

        atten_prob = torch.softmax(atten_scores_scaled, dim=-1)

        output = torch.matmul(atten_prob, value)

        return output

    def forward(self, input_data, mask=None, encoder_output=None):

        query_w = self.query_layer(input_data)

        if encoder_output is not None:
            key_w = self.key_layer(encoder_output)
            value_w = self.value_layer(encoder_output)
        else:
            key_w = self.key_layer(input_data)
            value_w = self.value_layer(input_data)

        attention_results = self.scaled_dot_product_attention(
            query_w, key_w, value_w, mask)

        attention_results = self.dropout(attention_results)

        return attention_results


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, num_heads, input_dim) -> None:
        super(MultiHeadAttentionWrapper, self).__init__()

        self.heads = nn.ModuleList([SelfAttention(input_dim)
                                   for _ in range(num_heads)])

        self.out_proj = nn.Linear(input_dim*num_heads, input_dim)

    def forward(self, input_data, attention_mask=None, encoder_output=None):

        attention_scores = [head.forward(
            input_data, mask=attention_mask, encoder_output=encoder_output) for head in self.heads]

        context_vec = torch.cat(attention_scores, dim=-1)

        return self.out_proj(context_vec)


class MultiHeadedAttention(nn.Module):

    def __init__(self, num_heads, input_dim, dropout=0.1) -> None:
        super(MultiHeadedAttention, self).__init__()

        self.num_heads = num_heads
        self.input_dim = input_dim
        self.query_layer = nn.Linear(input_dim, input_dim)
        self.key_layer = nn.Linear(input_dim, input_dim)
        self.value_layer = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim*num_heads, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_data, mask=None, encoder_output=None):

        query_w = torch.stack(
            [self.query_layer(input_data)]*self.num_heads, dim=0)

        if encoder_output is not None:
            key_w = torch.stack(
                [self.key_layer(encoder_output)]*self.num_heads, dim=0)
            value_w = torch.stack(
                [self.value_layer(encoder_output)]*self.num_heads, dim=0)
        else:
            key_w = torch.stack([self.key_layer(input_data)]
                                * self.num_heads, dim=0)
            value_w = torch.stack(
                [self.value_layer(input_data)]*self.num_heads, dim=0)

        attention_src = torch.matmul(query_w, key_w.transpose(-2, -1))

        attention_src = attention_src / \
            torch.sqrt(torch.tensor(self.input_dim))

        attention_prob = torch.softmax(attention_src, dim=-1)

        attention_output = torch.matmul(attention_prob, value_w)

        attention_output = self.dropout(attention_output)

        attention_scores = torch.cat(attention_output, dim=-1)

        return self.out_proj(attention_scores)