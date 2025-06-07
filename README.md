# GL-Fusion: Optimized Implementation for Human Mobility Prediction

This repository hosts a clean, optimized, and scalable implementation of the GL-Fusion model, designed for predicting human mobility patterns. The model leverages a hybrid approach, combining Graph Neural Networks (GNNs) to capture spatial dependencies and Large Language Models (LLMs) to understand sequential trajectory data. This implementation is tailored for efficient execution, particularly on High-Performance Computing (HPC) environments like NCI Gadi.

## 🚀 **Key Innovation: Enhanced Semantic Preprocessing**

A core innovation of this implementation is the **enhanced preprocessing pipeline**. Traditional models often treat locations as meaningless tokens (e.g., `node_123`), forcing the model to learn spatial relationships from scratch. This approach transforms raw spatial-temporal data into rich, semantically meaningful representations that the LLM can natively understand, dramatically improving learning efficiency and prediction quality.

-   **Before**: The model sees `"node_123 -> node_456"`.
-   **After**: The model sees a narrative like `"Journey starts on a weekday morning in a residential area, then moves to an area with professional services and dining options."`

This allows the model to leverage its pre-trained knowledge to reason about human behavior, such as recognizing commute patterns or trips for meals, which is crucial for solving the "mean prediction" problem where the model previously defaulted to a central location.

## 📊 Detailed Project Pipeline and Workflow

The project is structured as an end-to-end pipeline, from data ingestion to model evaluation, orchestrated by the `submit_single_node_enhanced.pbs` script.

```mermaid
graph TD
    subgraph "Phase 1: Preprocessing"
        A[Raw Data: Trajectories & POIs] --> B{run_enhanced_preprocessing_refined.py};
        B --> C[Enhanced Node Descriptions <br> (JSON)];
        B --> D[Trajectory Narratives <br> (PyTorch File)];
        B --> E[Spatial Graph <br> (PyTorch Geometric)];
    end

    subgraph "Phase 2: Training"
        F(Data Loader) --> G[GL-Fusion Model];
        C --> F;
        D --> F;
        E --> G;

        subgraph "GL-Fusion Model"
            H[GNN: Spatial Embeddings]
            I[LLM: Narrative Understanding]
            J[Fusion: Cross-Attention]
            H --> J;
            I --> J;
        end
        
        G --> K[Loss Calculation (MSE)];
        K --> L[Optimization <br> (DeepSpeed & AdamW)];
        L --> G;
    end

    subgraph "Phase 3: Evaluation"
        M[Load Best Checkpoint] --> N[Generate Predictions];
        N --> O{Inverse-Transform Coordinates};
        O --> P[Calculate Metrics <br> (MSE, GeoBLEU)];
    end
    
    L --> M;

```

### **Step 1: Enhanced Data Preprocessing**

This crucial first step transforms raw data into a format that leverages the LLM's language capabilities.

-   **Script**: `src/data/run_enhanced_preprocessing_refined.py`
-   **Inputs**:
    -   `task{1,2}_dataset_kotae.csv`: Raw user trajectory data.
    -   `cell_POIcat.csv`: POI counts per grid cell.
    -   `POI_datacategories.csv`: The names of the 86 real POI categories.
-   **Process & Justification**:
    1.  **Load Real POI Categories**: Instead of using generic categories, the script loads the **86 specific POI names** (e.g., "Japanese restaurant," "Electronics Store").
        -   **Justification**: This provides fine-grained, real-world context that is much more meaningful to an LLM than abstract categories.
    2.  **Generate Semantic Node Descriptions**: For each unique `(x, y)` location, a rich text description is generated.
        -   **Example**: `"Location at grid coordinates (120, 85). Primary functions: professional services, dining and food services. Features: 1 bank, 2 japanese restaurants, 1 post office."`
        -   **Justification**: This converts a simple coordinate into a semantic concept. The model no longer learns about "node 5432," but about a "business district with lunch options."
    3.  **Create Trajectory Narratives**: Each user's sequence of movements is converted into a natural language story.
        -   **Example**: `"Journey starts on weekday day 5 during early morning at an area with residential functions. Then moves to an area with professional services."`
        -   **Justification**: This frames the prediction task in a format perfectly suited for an LLM. The model can now use its understanding of language and common sense to recognize patterns like "commuting to work."
-   **Key Outputs**:
    -   `enhanced_node_descriptions_task2.json`: A mapping from each `node_id` to its rich text description.
    -   `enhanced_llm_sequences_task2.pt`: A file containing the token sequence, the full narrative, and other contextual data for each user trajectory.
    -   Standard outputs like `graph_data.pt` and data splits.

### **Step 2: Data Loading and Batching**

This stage efficiently prepares the enhanced data for the model.

-   **Scripts**: `src/training_v2/data/loader.py` and `src/training/utils.py`.
-   **Process & Justification**:
    1.  **Data Loader Creation**: The `DataLoaderFactory` is responsible for creating the `DataLoader` instances for the training and validation sets.
    2.  **Efficient Dataset Handling**: The `TrajectoryDataset` loads the `enhanced_llm_sequences_task2.pt` file. For memory efficiency, it creates "sample identifiers" rather than loading all narratives into RAM at once.
    3.  **Dynamic Batch Creation**: When a batch is requested by the training loop, the `collate_trajectories` function:
        -   Retrieves the **narrative text** for each sample in the batch.
        -   Uses the LLM's tokenizer to convert these narratives into `input_ids` and `attention_mask`.
        -   Pads the sequences to a uniform length within the batch.
        -   Assembles the final tensor dictionary required by the model.

### **Step 3: The GL-Fusion Model Architecture**

The core of the project, this model combines spatial and sequential understanding.

-   **Script**: `src/model/gl_fusion_model.py`
-   **Components & Justification**:
    1.  **GNN Path (Spatial Context)**:
        -   **Process**: A Graph Attention Network (GAT) processes the spatial graph where nodes are locations. It learns a "spatial embedding" for each location based on its own POI features and the features of its neighbors.
        -   **Justification**: The GNN understands the physical layout and functional properties of the environment, answering "What is this place like, and what is nearby?"
    2.  **LLM Path (Sequential & Semantic Context)**:
        -   **Process**: A Qwen2.5-7B model, efficiently adapted using **LoRA**, ingests the rich **trajectory narratives**.
        -   **Justification**: The LLM excels at understanding sequence, context, and the semantic meaning embedded in the narratives. It answers "Given this story of movement, what is the logical next chapter?"
    3.  **Fusion Module (Cross-Attention)**:
        -   **Process**: This is where the two streams merge. The contextualized output from the LLM (representing the "story") acts as a **query** to the GNN's spatial embeddings.
        -   **Justification**: This allows the model to dynamically decide which spatial information is most relevant for a given point in a trajectory. For a "commute to work" narrative, it will learn to pay more attention to the spatial embeddings of office buildings.
    4.  **Prediction Head**: A final Multi-Layer Perceptron (MLP) takes the fused representation and outputs the predicted `(x, y)` coordinates.

### **Step 4: Training and Optimization**

This stage handles the model learning process, optimized for large-scale HPC environments.

-   **Scripts**: `src/training_v2/main.py` and `src/training_v2/core/trainer.py`.
-   **Process & Justification**:
    1.  **Distributed Training**:
        -   **Mechanism**: The `torchrun` launcher and DeepSpeed library are used to manage training across multiple GPUs.
        -   **Justification**: This is essential for training a 7B-parameter model in a reasonable timeframe. DeepSpeed's ZeRO optimization significantly reduces memory requirements.
    2.  **Training Loop**:
        -   The `Trainer` class iterates through batches from the data loader.
        -   **Loss Calculation**: The model's coordinate predictions are compared against the ground truth using **Mean Squared Error (MSE)**.
        -   **Justification**: MSE is a standard and effective loss function for regression tasks like coordinate prediction.
    3.  **Checkpointing**: The model's state is periodically saved, with the best-performing version (based on validation loss) being retained for evaluation.

### **Step 5: Evaluation**

This final stage measures the model's performance.

-   **Script**: `src/evaluation/evaluate.py`
-   **Process & Justification**:
    1.  **Model Loading**: The script loads the best saved model checkpoint.
    2.  **Prediction Generation**: It generates predictions for each sample in the validation or test set.
    3.  **Inverse Transformation**: The model predicts coordinates scaled to a `[0, 1]` range. This step uses saved `MinMaxScaler` objects to convert these predictions back to their original grid coordinates (e.g., 0-200).
        -   **Justification**: This is a critical step to ensure the predicted coordinates are in the same space as the ground truth for accurate metric calculation.
    4.  **Metric Calculation**: It computes MSE, Euclidean distance, and leverages the provided `geobleu-2023` package to calculate the competition's primary metrics, GEO-BLEU and DTW.

## Repository Structure

```
GL/
├── config/
│   ├── model_config.yaml          # Main configuration file
│   └── deepspeed_config.json      # DeepSpeed config
├── data/
│   ├── raw/
│   ├── processed/
│   └── prepared_for_llm/
├── geobleu-2023/                  # GEO-BLEU evaluation scripts
├── results/
├── src/
│   ├── data/
│   │   ├── preprocess.py          # Standard preprocessing
│   │   └── run_enhanced_preprocessing_refined.py # Enhanced preprocessing script
│   ├── model/
│   │   └── gl_fusion_model.py     # Main GL-Fusion model
│   ├── training_v2/               # Modular training system
│   └── evaluation/
│       └── evaluate.py            # Evaluation script
├── *.pbs                          # PBS submission scripts
├── requirements.txt               # Python package dependencies
└── README.md                      # This file
```

## Getting Started

### Environment Setup (NCI Gadi Example)

1.  **Load Modules**:
    ```bash
    module load python3/3.9.2 cuda/12.3.2 gcc/11.1.0 openmpi/4.1.4
    ```

2.  **Activate Conda Environment**:
    ```bash
    source ~/.bashrc
    condapbs_ex gl
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Training with Enhanced Preprocessing

The `submit_single_node_enhanced.pbs` script automates the entire pipeline.

-   **Run Training**:
    ```bash
    qsub submit_single_node_enhanced.pbs
    ```

This script will automatically run the enhanced preprocessing if needed, then launch the training with the optimized configuration, and finally evaluate the best model.

## Configuration

The main configuration file is `config/model_config.yaml`. Key settings for the enhanced pipeline include:
-   **Data**: `train_data_subsample_ratio` (default is 0.5)
-   **LLM**: `lora_config` with optimized settings for Qwen2.5-7B
-   **Training**: `learning_rate`, `batch_size`, `gradient_accumulation_steps`

## Troubleshooting

### Out of Memory
- Reduce `batch_size` or increase `gradient_accumulation_steps` in your submission script.
- Ensure you are using the optimized LoRA and training configuration from `config/model_config.yaml`.

### Slow Data Loading
- The enhanced preprocessing is computationally intensive but only needs to be run once.
- Ensure you are using a sufficient number of CPUs in your PBS job for parallel processing.

## Citation

If you use this code or the GL-Fusion methodology, please consider citing the original GL-Fusion paper and acknowledge this optimized implementation.

## License

This project is licensed under the MIT License. 