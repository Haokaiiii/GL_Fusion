# GL-Fusion: Optimized Implementation for Human Mobility Prediction

This repository hosts a clean, optimized, and scalable implementation of the GL-Fusion model, designed for predicting human mobility patterns. The model leverages a hybrid approach, combining Graph Neural Networks (GNNs) to capture spatial dependencies and Large Language Models (LLMs) to understand sequential trajectory data. This implementation is tailored for efficient execution, particularly on High-Performance Computing (HPC) environments like NCI Gadi.

## Overview

The GL-Fusion model aims to predict future locations of individuals based on their historical movement trajectories and the surrounding Points of Interest (POI). Originally conceptualized for challenges like the Human Mobility Prediction Challenge (HuMob Challenge), this implementation focuses on:

-   **Clarity**: A simplified and modular architecture.
-   **Efficiency**: Memory-optimized operations, subgraph sampling for GNNs, and LoRA for LLM fine-tuning.
-   **Scalability**: Designed for both single-node and multi-node (multi-GPU) training setups.
-   **Maintainability**: A well-structured codebase with clear configurations.

## Project Workflow

The project follows a comprehensive pipeline from data processing to model evaluation:

1.  **Data Ingestion & Preprocessing (`src/data/preprocess.py`)**:
    *   Raw trajectory data (user, day, time, x, y coordinates) and POI information are loaded.
    *   Data is split into training, validation, and test sets.
    *   Spatial grid cells are mapped to node IDs.
    *   A spatial graph is constructed using PyTorch Geometric, with nodes representing grid cells and features derived from POIs.
    *   Textual descriptions for each graph node are generated from POI categories to be used by the LLM.
    *   Trajectory sequences are prepared in a format suitable for the LLM, incorporating special tokens and references to graph nodes.

2.  **Data Loading & Batching (`src/training_v2/data/trajectory_dataset.py`, `src/training_v2/data/custom_collate.py`)**:
    *   The `TrajectoryDataset` loads preprocessed graph data, node text descriptions, and LLM-ready sequences.
    *   A custom `collate_trajectories` function handles batching, including padding for variable-length sequences and assembling all necessary model inputs (LLM input IDs, attention masks, graph node IDs, node token indices, subgraph structures, and target coordinates).

3.  **Model Architecture (`src/model/gl_fusion_model.py`)**:
    The GL-Fusion model integrates three main components:
    *   **GNN Module (`src/model/gnns.py`)**: Processes the spatial graph structure (e.g., using Graph Attention Networks - GAT) to learn representations of locations based on their POI features and connectivity.
    *   **LLM Module (`src/model/llm_wrapper.py`)**: Utilizes a pre-trained Large Language Model (e.g., Qwen series) fine-tuned with LoRA. It processes the textualized trajectory sequences, including embedded graph node information.
    *   **Fusion Module (`src/model/fusion.py`)**: Employs mechanisms like cross-attention to effectively combine the spatial embeddings from the GNN and the sequential, contextual embeddings from the LLM. The architecture is designed to incorporate Structure-Aware Transformer layers and Graph-Text Cross-Attention.

4.  **Training (`src/training_v2/core/trainer.py`, `src/training_v2/main.py`)**:
    *   The training process is managed by a modular system, configurable via YAML files (`config/model_config_clean.yaml`).
    *   Supports distributed training using DeepSpeed for multi-GPU and multi-node setups.
    *   The training loop involves:
        *   Forward pass through the GL-Fusion model.
        *   Loss calculation (e.g., Mean Squared Error on predicted `(x, y)` coordinates).
        *   Backward pass and optimizer step.
    *   Metrics are logged (e.g., using TensorBoard), and model checkpoints are saved.

5.  **Evaluation (`src/evaluation/evaluate.py`)**:
    *   The best performing model checkpoint is loaded.
    *   Predictions are generated for the test set.
    *   Standard human mobility prediction metrics like GEO-BLEU and Dynamic Time Warping (DTW) are calculated using provided evaluation scripts (e.g., from `geobleu-2023/`).

6.  **Output Generation (`src/evaluation/generate_submission.py`)**:
    *   For challenges or specific deployment scenarios, predictions are formatted into the required CSV submission file format.

## Key Features & Optimizations

-   **Hybrid GNN-LLM Architecture**: Combines the strengths of GNNs for spatial reasoning and LLMs for sequential understanding.
-   **Subgraph Sampling**: GNN operates on relevant subgraphs for each batch, improving efficiency.
-   **LoRA Fine-Tuning**: Efficiently adapts large pre-trained LLMs for the specific task.
-   **Memory Efficiency**: Gradient checkpointing, 8-bit quantization (optional), and optimized tensor operations are employed.
-   **Modular Codebase**: Refactored into `src/training_v2/` for better maintainability and clarity.
-   **DeepSpeed Integration**: Robust support for distributed training, significantly speeding up experiments on HPC clusters.
-   **Flexible Configuration**: YAML-based system (`config/model_config_clean.yaml`) for managing all model, data, and training parameters.
-   **Environment Adaptability**: Scripts and code designed to work on NCI Gadi, including handling of environment modules and job submission.
-   **Data Subsampling**: Allows training on a fraction of the data for faster iteration and debugging.

## Repository Structure

```
GL/
├── config/
│   ├── model_config_clean.yaml    # Main configuration file
│   └── deepspeed_config.json      # Example DeepSpeed config
├── data/                          # Data (raw, processed, prepared_for_llm) - Gitignored
├── geobleu-2023/                  # GEO-BLEU evaluation scripts
├── project_plan/                  # Detailed project planning documents (Gitignored)
│   └── project.md                 # Main project plan, milestones, detailed design
├── results/                       # Outputs: checkpoints, logs, predictions (Gitignored)
├── src/
│   ├── config/                    # Configuration loading utilities
│   ├── data/                      # Data preprocessing scripts
│   ├── evaluation/                # Evaluation and submission generation scripts
│   ├── model/                     # Model component implementations
│   │   ├── gl_fusion_model.py     # Main GL-Fusion model
│   │   ├── fusion.py              # Cross-attention fusion
│   │   ├── llm_wrapper.py         # LLM wrapper
│   │   └── gnns.py                # GNN implementations
│   ├── training/                  # Original training scripts (being phased out)
│   ├── training_v2/               # New modular training system
│   │   ├── core/                  # Core training loop, distributed setup
│   │   ├── data/                  # Dataset, dataloader, collate functions
│   │   ├── distributed/           # Distributed training helpers
│   │   ├── models/                # Model factory
│   │   └── utils/                 # General utilities
│   └── utils/                     # Shared utility functions
├── *.pbs                          # PBS submission scripts (e.g., submit_single_node_dev.pbs)
├── requirements.txt               # Python package dependencies
└── README.md                      # This file
```

## Getting Started

### Environment Setup (NCI Gadi Example)

1.  **Access Gadi & Load Modules**:
    ```bash
    # Connect to Gadi
    ssh your_username@gadi.nci.org.au

    # Load necessary modules (adjust versions as per your environment)
    module load python3/3.9.2 # Or your project's python
    module load cuda/11.7     # Or a compatible CUDA version for your PyTorch
    module load openmpi/4.1.4 # For multi-node
    # Potentially others like cudnn, nccl
    ```

2.  **Conda Environment**:
    It is recommended to use a Conda environment. Ensure your environment is packed and transferred to Gadi if created elsewhere, or use a shared Conda installation. Activate using:
    ```bash
    source ~/.bashrc
    condapbs_ex <your_env_name>
    ```
    Or, if using a locally managed Conda:
    ```bash
    conda activate <your_env_name>
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: For DeepSpeed, ensure it's compiled correctly for the Gadi environment or use a pre-compiled version if available.*

### Training

The project includes PBS submission scripts for NCI Gadi. Adapt these scripts for your specific account, resource needs, and environment name.

-   **Single-Node Training**:
    Edit `submit_single_node_dev.pbs` (or a similar script) and submit:
    ```bash
    qsub submit_single_node_dev.pbs
    ```

-   **Multi-Node Training**:
    Edit `submit_multinode_dev.pbs` (or a similar script) and submit:
    ```bash
    qsub submit_multinode_dev.pbs
    ```

Key environment variables can often be set in the PBS script or on the command line to override defaults in `config/model_config_clean.yaml`:
```bash
# Example: Inside a PBS script or for a direct python call
export TASK_ID=1                  # Task 1 or 2 from HuMob Challenge
export BATCH_SIZE=4               # Per-GPU batch size
export GRADIENT_ACCUMULATION=8    # Gradient accumulation steps
export DATA_SUBSAMPLE_RATIO=0.1   # Use 10% of training data for quick tests
# export DS_SKIP_CUDA_CHECK=1     # If facing DeepSpeed CUDA check issues
# export DEEPSPEED_DISABLE_TRITON=1 # If Triton causes issues

# Then run your training command, e.g.:
# python src/training_v2/main.py --config_path config/model_config_clean.yaml
```

## Configuration

The main configuration file is `config/model_config_clean.yaml`. It controls:
-   **Data**: Paths, sequence lengths, subsampling ratios, number of workers.
-   **Model**: GNN architecture, LLM model name/path, LoRA settings, fusion parameters.
-   **Training**: Batch size, learning rate, epochs, gradient accumulation, optimizer settings.
-   **DeepSpeed**: Path to DeepSpeed config JSON.

## Current Status and Roadmap

This GL-Fusion implementation has undergone significant refactoring for clarity, modularity (`src/training_v2/`), and robust DeepSpeed integration. The core data processing, model structure, and training loop are functional.

**Current Focus:**
-   Fine-tuning and experimentation with the GL-Fusion architecture components (Structure-Aware Transformers, Graph-Text Cross-Attention).
-   Optimizing performance and scalability on multi-node HPC setups.

**Future Goals:**
-   Thorough evaluation on benchmark datasets like HuMob.
-   Exploration of advanced fusion techniques and model variants.

## Tips for NCI Gadi

-   **JobFS**: Utilize `$PBS_JOBFS` for fast temporary data caching (e.g., dataset caches) to speed up I/O.
-   **Resource Requests**: Adjust `#PBS` directives for `ncpus`, `mem`, `ngpus`, `walltime`, and `jobfs` based on your needs. Refer to `project_plan/project.md` and existing submission scripts for examples.
-   **Environment Variables**: Set `NCCL_SOCKET_IFNAME=ib0`, `NCCL_IB_DISABLE=0` for optimal InfiniBand usage with NCCL on Gadi V100 nodes. Consult NCI documentation for the latest recommendations for your specific GPU type.
-   **DeepSpeed on Gadi**: May require specific environment variables (e.g., `DS_SKIP_CUDA_CHECK=1`, `DS_DISABLE_TRITON=1`) and careful setup of compilers/CUDA toolkit. Patches for DeepSpeed (`src/patch_deepspeed.py`) might be necessary for certain versions.

## Troubleshooting

### Out of Memory
- Reduce `BATCH_SIZE` or increase `GRADIENT_ACCUMULATION` in your configuration or environment variables.
- Enable 8-bit quantization if not already (may require model adjustments).
- Reduce `max_edge_distance` or other complexity parameters in GNN configuration within `config/model_config_clean.yaml`.

### Multi-Node Issues
- Check SSH connectivity between nodes if using a manual MPI setup.
- Verify InfiniBand is working: `ibstat` on compute nodes.
- Ensure NCCL environment variables (`NCCL_SOCKET_IFNAME`, etc.) are correctly set and propagated.
- Check PBS job logs for errors related to MPI, DeepSpeed, or internode communication.

### Slow Data Loading
- Increase `num_workers` in `config/model_config_clean.yaml` for the DataLoader.
- Use `DATASET_CACHE_DIR=$PBS_JOBFS` in your submission script and ensure your `TrajectoryDataset` uses this for caching preprocessed items.

## Citation

If you use this code or the GL-Fusion methodology, please consider citing the original GL-Fusion paper (e.g., arXiv:2412.06849v1 as referenced in `project_plan/project.md`) and acknowledge this optimized implementation.

## License

This project is licensed under the MIT License. 