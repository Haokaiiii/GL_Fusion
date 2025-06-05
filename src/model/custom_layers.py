"""
Custom Transformer layers for GL-Fusion model.
Implements Structure-Aware Transformer Layers and Graph-Text Cross-Attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

# Setup logging
logger = logging.getLogger(__name__)


class StructureAwareTransformerLayer(nn.Module):
    """
    Structure-Aware Transformer Layer as described in the GL-Fusion paper.
    
    This layer integrates GNN message passing within the transformer architecture,
    including custom attention masking and a gating mechanism.
    """
    
    def __init__(self, hidden_dim, num_heads, gnn_model, dropout=0.1):
        """
        Initialize the Structure-Aware Transformer Layer.
        
        Args:
            hidden_dim (int): Hidden dimension of the transformer layer.
            num_heads (int): Number of attention heads.
            gnn_model (nn.Module): GNN model for message passing.
            dropout (float): Dropout probability.
        """
        super(StructureAwareTransformerLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.gnn_model = gnn_model
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # Gating mechanism for GNN integration
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, node_mask=None, attention_mask=None):
        """
        Forward pass through the Structure-Aware Transformer Layer.
        
        Args:
            x (torch.Tensor): Input token embeddings [batch_size, seq_len, hidden_dim].
            edge_index (torch.Tensor): Graph edge indices [2, num_edges].
            node_mask (torch.Tensor): Binary mask indicating which tokens are graph nodes [batch_size, seq_len].
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_len, seq_len].
            
        Returns:
            torch.Tensor: Output embeddings [batch_size, seq_len, hidden_dim].
        """
        batch_size, seq_len, _ = x.shape
        
        # Step 1: Apply self-attention with custom mask if provided
        # Normalize before attention (pre-norm)
        norm_x = self.norm1(x)
        
        # Apply self-attention
        attn_output, _ = self.self_attention(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=attention_mask
        )
        
        # Residual connection
        x = x + self.dropout(attn_output)
        
        # Step 2: Apply GNN message passing for node tokens
        if node_mask is not None:
            # Normalize before GNN (pre-norm)
            norm_x = self.norm2(x)
            
            # Create tensor to hold GNN outputs, initialized with zeros
            gnn_output = torch.zeros_like(norm_x)
            
            # For each item in the batch
            for b in range(batch_size):
                # Get indices of node tokens
                node_indices = torch.where(node_mask[b])[0]
                
                # Skip if no node tokens
                if len(node_indices) == 0:
                    continue
                
                # Extract node token embeddings
                node_embeddings = norm_x[b, node_indices]
                
                # Apply GNN message passing
                updated_node_embeddings = self.gnn_model(node_embeddings, edge_index)
                
                # Place updated embeddings back (using only the positions marked in node_mask)
                gnn_output[b, node_indices] = updated_node_embeddings
            
            # Gating mechanism (Eq. 2 in the paper)
            # Concatenate original and GNN-processed features
            concat_features = torch.cat([norm_x, gnn_output], dim=-1)
            gate_values = torch.sigmoid(self.gate(concat_features))
            
            # Element-wise gating
            gated_output = gate_values * gnn_output + (1 - gate_values) * norm_x
            
            # Residual connection with gated output
            x = x + self.dropout(gated_output)
        
        # Step 3: Apply feed-forward network
        # Normalize before FFN (pre-norm)
        norm_x = self.norm3(x)
        
        # Apply feed-forward network
        ffn_output = self.feed_forward(norm_x)
        
        # Residual connection
        x = x + self.dropout(ffn_output)
        
        return x


class GraphTextCrossAttention(nn.Module):
    """
    Graph-Text Cross-Attention as described in the GL-Fusion paper.
    
    This module allows attention between sequence tokens and uncompressed 
    node features/text.
    """
    
    def __init__(self, query_dim, key_dim, hidden_dim, num_heads, dropout=0.1):
        """
        Initialize the Graph-Text Cross-Attention.
        
        Args:
            query_dim (int): Dimension of query vectors (from sequence).
            key_dim (int): Dimension of key/value vectors (from graph).
            hidden_dim (int): Hidden dimension of the attention mechanism.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
        """
        super(GraphTextCrossAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        # Projection layers
        self.q_proj = nn.Linear(query_dim, hidden_dim)
        self.k_proj = nn.Linear(key_dim, hidden_dim)
        self.v_proj = nn.Linear(key_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, query_dim)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(query_dim)
        self.layer_norm2 = nn.LayerNorm(query_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim * 4, query_dim),
            nn.Dropout(dropout)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, sequence_hidden_states, node_text_states, attention_mask=None):
        """
        Forward pass through the Graph-Text Cross-Attention.
        
        Args:
            sequence_hidden_states (torch.Tensor): Hidden states from sequence [batch_size, seq_len, query_dim].
            node_text_states (torch.Tensor): States from node text [batch_size, num_nodes, key_dim].
            attention_mask (torch.Tensor, optional): Attention mask [batch_size, seq_len, num_nodes].
            
        Returns:
            torch.Tensor: Output embeddings [batch_size, seq_len, query_dim].
        """
        batch_size, seq_len, _ = sequence_hidden_states.shape
        _, num_nodes, _ = node_text_states.shape
        
        # Layer normalization before attention
        norm_sequence = self.layer_norm1(sequence_hidden_states)
        
        # Project queries, keys, and values
        q = self.q_proj(norm_sequence)
        k = self.k_proj(node_text_states)
        v = self.v_proj(node_text_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if attention_mask is not None:
            # Reshape mask for broadcasting
            mask = attention_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, v)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(context)
        
        # Residual connection
        output = sequence_hidden_states + self.dropout(output)
        
        # Layer normalization before FFN
        norm_output = self.layer_norm2(output)
        
        # Apply feed-forward network
        ffn_output = self.ffn(norm_output)
        
        # Residual connection
        output = output + self.dropout(ffn_output)
        
        return output 