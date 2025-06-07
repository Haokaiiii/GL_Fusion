"""
GL-Fusion model combining GNN, LLM, and fusion components for human mobility prediction.
Cleaned up and optimized version.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import logging
from typing import Dict, Optional, Tuple, List

from .gnns import create_gnn_model
from .llm_wrapper import QwenModel
from .fusion import CrossAttentionFusion

logger = logging.getLogger(__name__)


class GLFusionModel(nn.Module):
    """
    GL-Fusion model that combines GNN and LLM components with cross-attention fusion.
    Optimized for memory efficiency and cleaner code structure.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the GL-Fusion model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config
        
        # Initialize components
        self._init_llm()
        self._init_graph_components()
        self._init_fusion_and_prediction()
        
        logger.info(f"GLFusionModel initialized with {self.count_parameters():,} total parameters")
        logger.info(f"Trainable parameters: {self.count_parameters(trainable_only=True):,}")
    
    def _init_llm(self):
        """Initialize the LLM component."""
        self.llm = QwenModel(self.config)
        self.llm_hidden_dim = self.llm.get_hidden_dim()
        logger.info(f"LLM initialized with hidden dimension: {self.llm_hidden_dim}")
    
    def _init_graph_components(self):
        """Initialize graph-related components."""
        # Load node mapping
        node_mapping_path = self.config['data']['node_mapping']
        with open(node_mapping_path, 'r') as f:
            mapping_dict = json.load(f)
            self.node_mapping = {
                tuple(map(int, k.split(','))): v 
                for k, v in mapping_dict.items()
            }
        
        # Load graph data
        graph_data_path = self.config['data']['graph_data']
        graph_data = torch.load(graph_data_path, weights_only=False)
        
        # Register as buffers for automatic device handling
        self.register_buffer('node_features', graph_data.x.to(torch.float32))
        self.register_buffer('edge_index', graph_data.edge_index)
        
        # Feature projection layer
        num_node_features = self.node_features.shape[1]
        gnn_input_dim = self.config['gnn'].get('input_dim', 128)
        self.feature_projection = nn.Linear(num_node_features, gnn_input_dim)
        
        # Initialize GNN
        self.gnn = create_gnn_model(self.config['gnn'], gnn_input_dim)
        self.gnn_output_dim = self.config['gnn']['hidden_dim']
        
        logger.info(f"Graph components initialized: {self.node_features.shape[0]} nodes, "
                   f"{self.edge_index.shape[1]} edges")
    
    def _init_fusion_and_prediction(self):
        """Initialize fusion mechanism and prediction head."""
        fusion_config = self.config['fusion']
        
        self.fusion = CrossAttentionFusion(
            llm_hidden_size=self.llm_hidden_dim,
            node_hidden_size=self.gnn_output_dim,
            fusion_hidden_size=fusion_config['hidden_dim'],
            num_heads=fusion_config['num_heads'],
            dropout=fusion_config.get('dropout', 0.1)
        )
        
        # Temporal fusion for combining day and time tokens
        self.temporal_fusion = nn.MultiheadAttention(
            embed_dim=fusion_config['hidden_dim'],
            num_heads=fusion_config['num_heads'],
            dropout=fusion_config.get('dropout', 0.1),
            batch_first=True
        )
        
        # Temporal projection layer
        self.temporal_projection = nn.Linear(fusion_config['hidden_dim'], fusion_config['hidden_dim'])
        
        # Prediction head
        fusion_dim = fusion_config['hidden_dim']
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim // 2, 2)  # Predict (x, y) coordinates
        )
        
        # Initialize the final layer with smaller weights to prevent initial collapse
        # This encourages the model to start with diverse predictions
        with torch.no_grad():
            final_layer = self.predictor[-1]
            nn.init.normal_(final_layer.weight, mean=0.0, std=0.01)
            if final_layer.bias is not None:
                # Initialize bias to roughly center of scaled coordinates (0.5, 0.5)
                nn.init.constant_(final_layer.bias, 0.5)
        
        # Cache for graph embeddings to avoid recomputation
        self._cached_node_embeddings = None
        self._cached_node_ids = None
    
    def precompute_graph_embeddings(self):
        """Precompute embeddings for all nodes in the graph."""
        device = next(self.parameters()).device
        
        # Project all node features
        projected_features = self.feature_projection(self.node_features.to(device))
        
        # Process through GNN - keep in computational graph for training
        self._cached_node_embeddings = self.gnn(projected_features, self.edge_index.to(device))
        self._cached_node_ids = torch.arange(self.node_features.shape[0], device=device)
        
        logger.info(f"Precomputed embeddings for {self.node_features.shape[0]} nodes")
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        node_sequence_mapping: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        target_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the GL-Fusion model.
        
        Args:
            input_ids: LLM input token IDs [batch_size, seq_len]
            attention_mask: Attention mask for LLM [batch_size, seq_len]
            node_sequence_mapping: Mapping from sequence positions to node indices [batch_size, seq_len]
            node_mask: Binary mask indicating which tokens are graph nodes [batch_size, seq_len]
            target_positions: Target positions for loss calculation [batch_size, 2]
            
        Returns:
            Predicted coordinates [batch_size, 2] or loss if target_positions provided
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        # Create node mask if not provided
        if node_mask is None:
            node_token_id = self.llm.tokenizer.convert_tokens_to_ids('<node>')
            node_mask = (input_ids == node_token_id)
        
        # 1. Process graph with subgraph sampling for efficiency
        node_embeddings = self._process_graph_subgraph(node_sequence_mapping, device)
        
        # 2. Get LLM hidden states
        llm_hidden_states = self._get_llm_hidden_states(input_ids, attention_mask)
        
        # 3. Apply fusion mechanism
        fused_representation = self.fusion(
            llm_hidden_states, 
            node_embeddings, 
            node_sequence_mapping
        )
        
        # 4. Make predictions
        predictions = self._make_predictions(fused_representation, attention_mask, device)
        
        # 5. Calculate loss if targets provided and return consistent dictionary format
        if target_positions is not None:
            loss = self._calculate_loss(predictions, target_positions, device)
            return {
                'loss': loss,
                'predictions': predictions
            }
        
        # For inference, return predictions in dictionary format for consistency
        return {
            'predictions': predictions
        }
    
    def _process_graph_subgraph(
        self, 
        node_sequence_mapping: torch.Tensor, 
        device: torch.device
    ) -> torch.Tensor:
        """Process only the relevant subgraph for efficiency."""
        # Extract unique nodes from the batch
        unique_nodes = torch.unique(
            node_sequence_mapping[node_sequence_mapping >= 0]
        )
        
        if len(unique_nodes) == 0:
            # No valid nodes, return empty embeddings
            return torch.zeros(0, self.gnn_output_dim, device=device)
        
        # Use cached embeddings if available
        if self._cached_node_embeddings is not None:
            # Retrieve from cache. Move unique_nodes to CPU for indexing
            # Don't use .clone() as it might break gradient flow
            node_embeddings = self._cached_node_embeddings[unique_nodes.to('cpu')].to(device)
            
            # Create node ID mapping for compatibility
            node_id_map = {nid.item(): i for i, nid in enumerate(unique_nodes)}
            self._current_node_id_map = node_id_map
            
            return node_embeddings
        
        # Otherwise, compute embeddings on the fly (fallback for backward compatibility)
        unique_nodes_cpu = unique_nodes.cpu()
        
        # Move features to device and project
        subgraph_features = self.node_features[unique_nodes_cpu].to(device)
        projected_features = self.feature_projection(subgraph_features)
        
        # Ensure edge_index is on the correct device
        current_edge_index = self.edge_index.to(device)
        
        # Extract relevant edges
        edge_mask = torch.isin(current_edge_index[0], unique_nodes) & \
                   torch.isin(current_edge_index[1], unique_nodes)
        subgraph_edges = current_edge_index[:, edge_mask]
        
        # Create node ID mapping for the subgraph
        node_id_map = {nid.item(): i for i, nid in enumerate(unique_nodes)}
        
        # Remap edges to local indices
        if subgraph_edges.shape[1] > 0:
            remapped_edges = torch.zeros_like(subgraph_edges)
            for i in range(2):
                for j, node_id in enumerate(subgraph_edges[i]):
                    remapped_edges[i, j] = node_id_map.get(node_id.item(), 0)
            subgraph_edges = remapped_edges
        
        # Process through GNN
        node_embeddings = self.gnn(projected_features, subgraph_edges)
        
        # Store the mapping for later use
        self._current_node_id_map = node_id_map
        
        return node_embeddings
    
    def _get_llm_hidden_states(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get hidden states from the LLM."""
        llm_output = self.llm(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            labels=None
        )
        
        if hasattr(llm_output, 'hidden_states') and llm_output.hidden_states:
            return llm_output.hidden_states[-1]
        else:
            raise ValueError("LLM output does not contain hidden_states")
    
    def _make_predictions(
        self, 
        fused_representation: torch.Tensor,
        attention_mask: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """Make coordinate predictions from fused representations with temporal awareness."""
        batch_size = fused_representation.shape[0]
        seq_len = fused_representation.shape[1]
        
        # Get the last two positions (day and time tokens)
        # We assume the input format is [...history..., day_token, time_token]
        last_positions = attention_mask.sum(dim=1) - 1
        day_positions = last_positions - 1
        
        indices = torch.arange(batch_size, device=device)
        
        # Extract day and time representations
        day_representations = fused_representation[indices, day_positions].clone()
        time_representations = fused_representation[indices, last_positions].clone()
        
        # Also get the last node representation (before temporal tokens)
        # Find the last node position (3 positions before the end)
        node_positions = torch.clamp(last_positions - 2, min=0)
        last_node_representations = fused_representation[indices, node_positions].clone()
        
        # Stack temporal representations for attention
        temporal_features = torch.stack([day_representations, time_representations, last_node_representations], dim=1)
        
        # Apply temporal fusion to combine day, time, and last location context
        fused_temporal, _ = self.temporal_fusion(
            temporal_features, temporal_features, temporal_features
        )
        
        # Use the first (query) position instead of mean to preserve information
        # This represents the attended temporal context
        final_representation = fused_temporal[:, 0, :]
        
        # Apply temporal projection
        final_representation = self.temporal_projection(final_representation)
        
        # Apply predictor
        predictions = self.predictor(final_representation)
        
        return predictions
    
    def _calculate_loss(
        self, 
        predictions: torch.Tensor,
        target_positions: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """Calculate MSE loss for coordinate prediction."""
        targets = target_positions.to(device, dtype=predictions.dtype)
        return F.mse_loss(predictions, targets)
    
    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def predict_coordinates(
        self, 
        user_trajectory: List[Tuple[int, int, float, float]], 
        num_steps: int = 1
    ) -> List[Tuple[float, float]]:
        """
        Predict future coordinates for a user trajectory.
        
        Args:
            user_trajectory: List of (d, t, x, y) tuples
            num_steps: Number of future steps to predict
            
        Returns:
            List of predicted (x, y) coordinates
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Convert trajectory to node IDs
        node_ids = []
        for _, _, x, y in user_trajectory:
            node_id = self.node_mapping.get((int(x), int(y)), -1)
            if node_id != -1:
                node_ids.append(node_id)
        
        if not node_ids:
            logger.warning("No valid nodes in trajectory")
            return [(0.0, 0.0)] * num_steps
        
        # Encode sequence
        inputs = self.llm.encode_sequence(node_ids)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # Create node sequence mapping
        node_sequence_mapping = torch.tensor(
            [node_ids], dtype=torch.long
        ).to(device)
        
        # Adjust mapping size if needed
        seq_len = input_ids.shape[1]
        if node_sequence_mapping.shape[1] != seq_len:
            # Pad or truncate
            if node_sequence_mapping.shape[1] > seq_len:
                node_sequence_mapping = node_sequence_mapping[:, :seq_len]
            else:
                padding = seq_len - node_sequence_mapping.shape[1]
                node_sequence_mapping = F.pad(
                    node_sequence_mapping, (0, padding), value=-1
                )
        
        # Make predictions
        predictions = []
        with torch.no_grad():
            coords = self(
                input_ids,
                attention_mask,
                node_sequence_mapping
            )
            pred_coord = coords[0].cpu().numpy()
            predictions.append(tuple(pred_coord))
        
        # For multi-step prediction, we'd need to update the trajectory
        # and predict iteratively. For now, just repeat the prediction.
        return predictions * num_steps 