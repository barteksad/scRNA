import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from omegaconf import DictConfig
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel
from hydra.utils import instantiate
import os
import pickle
from pathlib import Path
from torch.utils.data import random_split


def setup_distributed():
    """Initialize distributed training"""
    # torchrun sets these environment variables
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Initialize the process group
    dist.init_process_group("nccl")

    # Set the device for this process
    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()


def is_main_process(rank=None):
    """Check if this is the main process (rank 0)"""
    if rank is not None:
        return rank == 0
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank():
    """Get the rank of the current process"""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    """Get the world size"""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


class GenomicsCLIP(nn.Module):
    def __init__(
        self,
        # Genomics encoder config
        cell_vocab_size: int = 57000,
        max_cell_tokens: int = 1200,
        cell_embed_dim: int = 512,
        cell_transformer_heads: int = 8,
        cell_transformer_layers: int = 4,
        # Text encoder config
        text_model_name: str = "google-bert/bert-base-cased",
        max_text_tokens: int = 128,
        text_proj_dim: int = 256,
        # Projection config
        projection_dim: int = 256,
        dropout: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()
        self.max_cell_tokens = max_cell_tokens
        self.max_text_tokens = max_text_tokens

        # ============= Genomics Encoder =============
        self.cell_embedding = nn.Embedding(cell_vocab_size, cell_embed_dim)
        self.cell_pos_embedding = nn.Parameter(
            torch.randn(1, max_cell_tokens, cell_embed_dim)
        )

        cell_encoder_layers = TransformerEncoderLayer(
            d_model=cell_embed_dim,
            nhead=cell_transformer_heads,
            dim_feedforward=cell_embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.cell_encoder = TransformerEncoder(
            cell_encoder_layers, cell_transformer_layers
        )

        # ============= Text Encoder =============
        self.text_encoder = BertModel.from_pretrained(text_model_name)

        # Freeze BERT
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # ============= Projection Heads =============
        self.cell_proj = nn.Sequential(
            nn.Linear(cell_embed_dim, projection_dim),
            nn.GELU(),
            nn.LayerNorm(projection_dim),
            nn.Linear(projection_dim, projection_dim),
        )

        self.text_proj = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, text_proj_dim),
            nn.GELU(),
            nn.LayerNorm(text_proj_dim),
            nn.Linear(text_proj_dim, projection_dim),
        )

        # Temperature parameter
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1 / 0.07))
        )

        self.device = device

    def encode_cells(self, cell_tokens: torch.Tensor) -> torch.Tensor:
        """Process tokenized cell data through genomics encoder"""
        # cell_tokens shape: (batch_size, seq_len)
        x = self.cell_embedding(cell_tokens)  # (batch, seq, embed)
        x = x + self.cell_pos_embedding[:, : cell_tokens.size(1), :]

        # Generate padding mask
        padding_mask = cell_tokens == 0

        x = self.cell_encoder(x, src_key_padding_mask=padding_mask)

        # Global average pooling
        x = x.mean(dim=1)  # (batch, embed)
        return x

    def encode_text(self, text_tokens: dict[str, torch.Tensor]) -> torch.Tensor:
        """Process tokenized text through BERT"""
        outputs = self.text_encoder(
            input_ids=text_tokens["input_ids"],
            attention_mask=text_tokens["attention_mask"],
        )
        # Use [CLS] token representation
        return outputs.last_hidden_state[:, 0, :]

    def forward(self, batch: dict[str, list]) -> tuple[torch.Tensor, torch.Tensor]:
        cell_tokens = batch["cell_tokens"].to(self.device)
        text_tokens = batch["input_ids"].to(self.device)
        attention_masks = batch["attention_mask"].to(self.device)

        cell_features = self.encode_cells(cell_tokens)
        text_features = self.encode_text(
            {"input_ids": text_tokens, "attention_mask": attention_masks}
        )

        # Project to joint space
        cell_embeddings = self.cell_proj(cell_features)
        text_embeddings = self.text_proj(text_features)

        # Normalize features
        cell_embeddings = cell_embeddings / cell_embeddings.norm(dim=1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

        # Cosine similarity with temperature
        logit_scale = self.logit_scale.exp()
        logits_per_cell = logit_scale * cell_embeddings @ text_embeddings.t()
        logits_per_text = logits_per_cell.t()

        return logits_per_cell, logits_per_text

    def compute_embeddings(
        self, dataloader: DataLoader, use_cache: bool = False, cache_dir: str = "cache"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute embeddings for dataset with proper distributed handling.

        In distributed training, each process computes embeddings for its subset
        and only allocates memory for the data it actually processes.
        """
        cache_path = Path(cache_dir)
        cache_path.mkdir(exist_ok=True)

        # Create cache file names based on model state and rank
        model_hash = hash(str(self.state_dict()))
        rank = get_rank()
        world_size = get_world_size()

        # For distributed training, use rank-specific cache files
        if world_size > 1:
            cell_cache_file = (
                cache_path / f"cell_embeddings_{model_hash}_rank_{rank}.pkl"
            )
            text_cache_file = (
                cache_path / f"text_embeddings_{model_hash}_rank_{rank}.pkl"
            )
        else:
            cell_cache_file = cache_path / f"cell_embeddings_{model_hash}.pkl"
            text_cache_file = cache_path / f"text_embeddings_{model_hash}.pkl"

        # Try to load from cache if enabled
        if use_cache and cell_cache_file.exists() and text_cache_file.exists():
            if is_main_process():
                print("Loading embeddings from cache...")
            with open(cell_cache_file, "rb") as f:
                cell_embeddings = pickle.load(f)
            with open(text_cache_file, "rb") as f:
                text_embeddings = pickle.load(f)
            return cell_embeddings, text_embeddings

        if is_main_process():
            print("Computing embeddings for dataset...")
        self.eval()

        # Calculate the number of samples this process will actually handle
        # In distributed training, this is only a subset of the full dataset
        total_batches = len(dataloader)
        if is_main_process():
            print(f"Processing {total_batches} batches on rank {rank}/{world_size}")

        # Collect embeddings in a list first to avoid pre-allocation issues
        cell_embeddings_list = []
        text_embeddings_list = []

        projection_dim = self.cell_proj[-1].out_features

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(
                    dataloader,
                    desc="Computing embeddings",
                    disable=not is_main_process(),
                )
            ):
                cell_tokens = batch["cell_tokens"].to(self.device)
                text_tokens = batch["input_ids"].to(self.device)
                attention_masks = batch["attention_mask"].to(self.device)

                # Encode modalities
                cell_features = self.encode_cells(cell_tokens)
                text_features = self.encode_text(
                    {"input_ids": text_tokens, "attention_mask": attention_masks}
                )

                # Project to joint space
                batch_cell_embeddings = self.cell_proj(cell_features)
                batch_text_embeddings = self.text_proj(text_features)

                # Normalize features
                batch_cell_embeddings = (
                    batch_cell_embeddings
                    / batch_cell_embeddings.norm(dim=1, keepdim=True)
                )
                batch_text_embeddings = (
                    batch_text_embeddings
                    / batch_text_embeddings.norm(dim=1, keepdim=True)
                )

                # Store embeddings (move to CPU to save GPU memory)
                cell_embeddings_list.append(batch_cell_embeddings.cpu())
                text_embeddings_list.append(batch_text_embeddings.cpu())

        # Concatenate all embeddings
        cell_embeddings = torch.cat(cell_embeddings_list, dim=0)
        text_embeddings = torch.cat(text_embeddings_list, dim=0)

        if is_main_process():
            print(
                f"Computed embeddings shape: cell={cell_embeddings.shape}, text={text_embeddings.shape}"
            )

        # Cache embeddings if enabled
        if use_cache:
            if is_main_process():
                print("Caching embeddings...")
            with open(cell_cache_file, "wb") as f:
                pickle.dump(cell_embeddings, f)
            with open(text_cache_file, "wb") as f:
                pickle.dump(text_embeddings, f)

        return cell_embeddings, text_embeddings

    def evaluate_whole_dataset_distributed(
        self, dataloader: DataLoader, use_cache: bool = False, cache_dir: str = "cache"
    ) -> dict[str, float]:
        """
        Evaluate model performance across the entire dataset using distributed approach.

        Each process computes embeddings for its subset, then all embeddings are
        gathered on the main process for global evaluation.
        """
        world_size = get_world_size()
        rank = get_rank()

        if is_main_process():
            print(f"Starting distributed evaluation with {world_size} processes")

        try:
            # Compute embeddings for this process's subset
            cell_embeddings, text_embeddings = self.compute_embeddings(
                dataloader, use_cache, cache_dir
            )

            if is_main_process():
                print(
                    f"Computed local embeddings: cell={cell_embeddings.shape}, text={text_embeddings.shape}"
                )

            if world_size == 1:
                # Single GPU case - compute metrics directly
                return self._compute_retrieval_metrics(cell_embeddings, text_embeddings)

            # Distributed case - gather all embeddings on main process
            if is_main_process():
                print("Gathering embeddings from all processes...")

            # Gather embeddings from all processes
            if dist.is_initialized():
                try:
                    # Convert to tensors and move to GPU for gathering
                    cell_embeddings_gpu = cell_embeddings.to(self.device)
                    text_embeddings_gpu = text_embeddings.to(self.device)

                    # Gather sizes first
                    local_size = torch.tensor(
                        [cell_embeddings.size(0)], device=self.device
                    )
                    all_sizes = [
                        torch.zeros_like(local_size) for _ in range(world_size)
                    ]
                    dist.all_gather(all_sizes, local_size)

                    if is_main_process():
                        print(
                            f"Gathered sizes from all processes: {[s.item() for s in all_sizes]}"
                        )

                    # Gather embeddings
                    all_cell_embeddings = []
                    all_text_embeddings = []

                    for i in range(world_size):
                        size = all_sizes[i].item()
                        if size > 0:
                            if i == rank:
                                all_cell_embeddings.append(cell_embeddings_gpu)
                                all_text_embeddings.append(text_embeddings_gpu)
                            else:
                                cell_placeholder = torch.zeros(
                                    size, cell_embeddings.size(1), device=self.device
                                )
                                text_placeholder = torch.zeros(
                                    size, text_embeddings.size(1), device=self.device
                                )
                                all_cell_embeddings.append(cell_placeholder)
                                all_text_embeddings.append(text_placeholder)

                    # Perform the actual gathering
                    for i in range(world_size):
                        if all_sizes[i].item() > 0:
                            dist.broadcast(all_cell_embeddings[i], src=i)
                            dist.broadcast(all_text_embeddings[i], src=i)

                    # Concatenate all embeddings on main process
                    if is_main_process():
                        global_cell_embeddings = torch.cat(
                            all_cell_embeddings, dim=0
                        ).cpu()
                        global_text_embeddings = torch.cat(
                            all_text_embeddings, dim=0
                        ).cpu()

                        print(
                            f"Global embeddings shape: cell={global_cell_embeddings.shape}, text={global_text_embeddings.shape}"
                        )

                        # Compute metrics on the full dataset
                        metrics = self._compute_retrieval_metrics(
                            global_cell_embeddings, global_text_embeddings
                        )
                        return metrics
                    else:
                        # Non-main processes return empty metrics
                        return {}

                except Exception as e:
                    if is_main_process():
                        print(f"Error in distributed evaluation: {e}")
                        print("Falling back to per-process evaluation...")
                    # Fallback to per-process evaluation
                    return self._compute_retrieval_metrics(
                        cell_embeddings, text_embeddings
                    )
            else:
                # Fallback if distributed not initialized
                return self._compute_retrieval_metrics(cell_embeddings, text_embeddings)

        except Exception as e:
            if is_main_process():
                print(f"Critical error in distributed evaluation: {e}")
                print("Returning empty metrics...")
            return {}

    def evaluate_whole_dataset_simple_distributed(
        self, dataloader: DataLoader
    ) -> dict[str, float]:
        """
        Simple distributed evaluation that computes metrics per process and averages them.
        This is a fallback method that's more memory efficient but less accurate.
        """
        world_size = get_world_size()
        rank = get_rank()

        if is_main_process():
            print(f"Starting simple distributed evaluation with {world_size} processes")

        # Compute embeddings for this process's subset
        cell_embeddings, text_embeddings = self.compute_embeddings(dataloader)

        # Compute metrics on local subset
        local_metrics = self._compute_retrieval_metrics(
            cell_embeddings, text_embeddings
        )

        if world_size == 1:
            return local_metrics

        # Reduce metrics across all processes
        if dist.is_initialized():
            try:
                # Convert metrics to tensors for reduction
                metric_names = [
                    "text_to_cell_accuracy",
                    "cell_to_text_accuracy",
                    "average_accuracy",
                ]
                reduced_metrics = {}

                for metric_name in metric_names:
                    if metric_name in local_metrics:
                        metric_tensor = torch.tensor(
                            local_metrics[metric_name], device=self.device
                        )
                        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
                        reduced_metrics[metric_name] = metric_tensor.item() / world_size

                # Handle top-k metrics
                for k in [1, 5, 10]:
                    for direction in ["text_to_cell", "cell_to_text"]:
                        metric_name = f"{direction}_top{k}_accuracy"
                        if metric_name in local_metrics:
                            metric_tensor = torch.tensor(
                                local_metrics[metric_name], device=self.device
                            )
                            dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
                            reduced_metrics[metric_name] = (
                                metric_tensor.item() / world_size
                            )

                return reduced_metrics

            except Exception as e:
                if is_main_process():
                    print(f"Error in simple distributed evaluation: {e}")
                return local_metrics

        return local_metrics

    def evaluate_whole_dataset_memory_efficient(
        self, dataloader: DataLoader, chunk_size: int = 1000
    ) -> dict[str, float]:
        """
        Memory-efficient evaluation that doesn't store all embeddings at once.
        Computes similarities in chunks to avoid OOM errors.
        """
        if get_world_size() > 1:
            if is_main_process():
                print(
                    "Memory-efficient evaluation not fully supported in distributed mode."
                )
                print("Falling back to subset evaluation per process.")

            # In distributed mode, each process evaluates its subset
            cell_embeddings, text_embeddings = self.compute_embeddings(dataloader)
            return self._compute_retrieval_metrics(cell_embeddings, text_embeddings)

        # Single GPU memory-efficient evaluation
        if is_main_process():
            print("Computing memory-efficient whole dataset evaluation...")

        self.eval()

        # Store embeddings in chunks
        all_cell_embeddings = []
        all_text_embeddings = []

        with torch.no_grad():
            for batch in tqdm(
                dataloader, desc="Computing embeddings", disable=not is_main_process()
            ):
                cell_tokens = batch["cell_tokens"].to(self.device)
                text_tokens = batch["input_ids"].to(self.device)
                attention_masks = batch["attention_mask"].to(self.device)

                # Encode modalities
                cell_features = self.encode_cells(cell_tokens)
                text_features = self.encode_text(
                    {"input_ids": text_tokens, "attention_mask": attention_masks}
                )

                # Project and normalize
                batch_cell_embeddings = self.cell_proj(cell_features)
                batch_text_embeddings = self.text_proj(text_features)

                batch_cell_embeddings = (
                    batch_cell_embeddings
                    / batch_cell_embeddings.norm(dim=1, keepdim=True)
                )
                batch_text_embeddings = (
                    batch_text_embeddings
                    / batch_text_embeddings.norm(dim=1, keepdim=True)
                )

                all_cell_embeddings.append(batch_cell_embeddings.cpu())
                all_text_embeddings.append(batch_text_embeddings.cpu())

        # Concatenate all embeddings
        cell_embeddings = torch.cat(all_cell_embeddings, dim=0)
        text_embeddings = torch.cat(all_text_embeddings, dim=0)

        return self._compute_retrieval_metrics(cell_embeddings, text_embeddings)

    def _compute_retrieval_metrics(
        self, cell_embeddings: torch.Tensor, text_embeddings: torch.Tensor
    ) -> dict[str, float]:
        """
        Compute retrieval metrics given cell and text embeddings.
        Uses chunked similarity computation to handle large datasets.
        """
        if is_main_process():
            print("Computing retrieval metrics...")

        # Compute similarity matrix in chunks to handle memory constraints
        similarity_matrix = self._compute_similarity_matrix_chunked(
            cell_embeddings, text_embeddings, chunk_size=1000
        )

        # Compute metrics
        metrics = {}

        # Text-to-cell retrieval (for each text, find best matching cell)
        text_to_cell_predictions = similarity_matrix.argmax(dim=1)
        text_to_cell_ground_truth = torch.arange(len(text_embeddings))
        text_to_cell_acc = (
            (text_to_cell_predictions == text_to_cell_ground_truth)
            .float()
            .mean()
            .item()
        )

        # Cell-to-text retrieval (for each cell, find best matching text)
        cell_to_text_predictions = similarity_matrix.argmax(dim=0)
        cell_to_text_ground_truth = torch.arange(len(cell_embeddings))
        cell_to_text_acc = (
            (cell_to_text_predictions == cell_to_text_ground_truth)
            .float()
            .mean()
            .item()
        )

        # Top-k accuracies
        for k in [1, 5, 10]:
            if k <= similarity_matrix.size(1):
                # Text-to-cell top-k
                _, text_to_cell_topk = similarity_matrix.topk(k, dim=1)
                text_to_cell_topk_acc = (
                    (text_to_cell_topk == text_to_cell_ground_truth.unsqueeze(1))
                    .any(dim=1)
                    .float()
                    .mean()
                    .item()
                )
                metrics[f"text_to_cell_top{k}_accuracy"] = text_to_cell_topk_acc

                # Cell-to-text top-k
                _, cell_to_text_topk = similarity_matrix.topk(k, dim=0)
                cell_to_text_topk_acc = (
                    (cell_to_text_topk == cell_to_text_ground_truth.unsqueeze(0))
                    .any(dim=0)
                    .float()
                    .mean()
                    .item()
                )
                metrics[f"cell_to_text_top{k}_accuracy"] = cell_to_text_topk_acc

        metrics.update(
            {
                "text_to_cell_accuracy": text_to_cell_acc,
                "cell_to_text_accuracy": cell_to_text_acc,
                "average_accuracy": (text_to_cell_acc + cell_to_text_acc) / 2,
            }
        )

        return metrics

    def evaluate_whole_dataset(
        self,
        dataloader: DataLoader,
        use_cache: bool = False,
        cache_dir: str = "cache",
        distributed_strategy: str = "gather",
    ) -> dict[str, float]:
        """
        Main evaluation method that chooses the appropriate evaluation strategy.

        Args:
            dataloader: DataLoader for the dataset
            use_cache: Whether to use embedding caching
            cache_dir: Directory for caching embeddings
            distributed_strategy: Strategy for distributed evaluation:
                - "gather": Gather all embeddings on main process (more accurate but memory intensive)
                - "reduce": Reduce metrics across processes (memory efficient but less accurate)
                - "memory_efficient": Use memory-efficient single-GPU evaluation
        """
        world_size = get_world_size()

        if world_size > 1:
            if distributed_strategy == "gather":
                # Use distributed evaluation for multi-GPU
                return self.evaluate_whole_dataset_distributed(
                    dataloader, use_cache, cache_dir
                )
            elif distributed_strategy == "reduce":
                # Use simple distributed evaluation (metric averaging)
                return self.evaluate_whole_dataset_simple_distributed(dataloader)
            else:
                # Fallback to memory-efficient evaluation
                return self.evaluate_whole_dataset_memory_efficient(dataloader)
        else:
            # Use memory-efficient evaluation for single GPU
            return self.evaluate_whole_dataset_memory_efficient(dataloader)

    def _compute_similarity_matrix_chunked(
        self,
        cell_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        chunk_size: int = 1000,
    ) -> torch.Tensor:
        """
        Compute similarity matrix in chunks to handle memory constraints.
        """
        num_texts = text_embeddings.size(0)
        num_cells = cell_embeddings.size(0)

        # Initialize similarity matrix
        similarity_matrix = torch.zeros(num_texts, num_cells)

        # Compute similarities in chunks
        for i in range(0, num_texts, chunk_size):
            end_i = min(i + chunk_size, num_texts)
            text_chunk = text_embeddings[i:end_i]  # (chunk_size, embedding_dim)

            # Compute similarity between text chunk and all cells
            chunk_similarities = (
                text_chunk @ cell_embeddings.t()
            )  # (chunk_size, num_cells)
            similarity_matrix[i:end_i] = chunk_similarities

        return similarity_matrix

    def predict_similarity_matrix(self, batch: dict[str, list]) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            _, logits_per_text = self.forward(batch)
            similarity_matrix = logits_per_text / self.logit_scale.exp()

        return similarity_matrix

    def predict_best_matches(self, batch: dict[str, list]) -> torch.Tensor:
        similarity_matrix = self.predict_similarity_matrix(batch)
        return similarity_matrix.argmax(dim=1)

    def accuracy_paired_batch(self, batch: dict[str, list]) -> float:
        assert len(batch["cell_tokens"]) == len(
            batch["input_ids"]
        )  # here we assume the batch contains paired text-cells
        y_hat = self.predict_best_matches(batch)
        y_true = torch.arange(len(batch["cell_tokens"]), device=self.device)

        return self.accuracy(y_hat, y_true)

    @staticmethod
    def accuracy(y_hat: torch.Tensor, y_true: torch.Tensor) -> float:
        return (y_hat == y_true).float().mean().item()

    # def tokenize_text(self, text):
    #     """Tokenize raw text"""
    #     encoding = self.text_tokenizer(
    #         text,
    #         return_tensors="pt",
    #         padding="max_length",
    #         truncation=True,
    #         max_length=self.max_text_tokens,
    #     )
    #     return encoding["input_ids"][0].to(self.device), encoding["attention_mask"][
    #         0
    #     ].to(self.device)

    # def tokenize_cells(self, cell_data):
    #     """Tokenize raw cell data"""
    #     x, obs, var = cell_data
    #     _, tokenized_cells = self.cell_tokenizer.tokenize_single_cell(x, obs, var)
    #     positional_encoding = torch.tensor(
    #         tokenized_cells[0].fillna(0).astype(int).values,
    #         dtype=torch.int32,
    #         device=self.device,
    #     )

    #     # Pad/truncate
    #     cell_tokens = torch.zeros(
    #         self.max_cell_tokens, dtype=torch.int32, device=self.device
    #     )
    #     actual_length = min(len(positional_encoding), self.max_cell_tokens)
    #     cell_tokens[:actual_length] = positional_encoding[:actual_length]
    #     return cell_tokens


def get_components(config):
    dataset = instantiate(config.dataset)

    return dataset


def train_clip(config: DictConfig):
    """Main training function that handles distributed setup"""
    # Check if we're in a distributed environment
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        # Multi-GPU distributed training
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Initialize distributed training
        rank, local_rank, world_size = setup_distributed()

        # Set device to local rank
        device = f"cuda:{local_rank}"

        train_clip_distributed(config, rank, world_size, device)

        # Cleanup
        cleanup_distributed()
    else:
        # Single GPU training
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_clip_single(config, device)


def train_clip_single(config: DictConfig, device: str):
    """Single GPU training (original logic)"""
    print("Starting single GPU CLIP training...")
    dataset = get_components(config)

    # Split dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Training set size: {train_size}, Validation set size: {val_size}")

    # Create model with parameters from config
    clip_model = GenomicsCLIP(
        cell_vocab_size=config.exp.cell_vocab_size,
        max_cell_tokens=config.exp.max_cell_tokens,
        cell_embed_dim=config.exp.cell_embed_dim,
        cell_transformer_heads=config.exp.cell_transformer_heads,
        cell_transformer_layers=config.exp.cell_transformer_layers,
        text_model_name=config.exp.text_model_name,
        max_text_tokens=config.exp.max_text_tokens,
        text_proj_dim=config.exp.text_proj_dim,
        projection_dim=config.exp.projection_dim,
        dropout=config.exp.dropout,
        device=device,
    )

    clip_model.to(device)

    # Extract training config parameters
    train_config = {
        "batch_size": config.exp.batch_size,
        "epochs": config.exp.epochs,
        "lr": config.exp.lr,
        "min_lr": config.exp.min_lr,
        "weight_decay": config.exp.weight_decay,
        "num_workers": config.exp.num_workers,
        "model_save_path": config.exp.model_save_path,
        "use_wandb": config.exp.use_wandb,
        "log_accuracy": config.exp.log_accuracy,
        "wandb_project": config.wandb.project,
        "wandb_entity": config.wandb.entity,
        "log_dir": config.exp.log_dir,
        "use_whole_dataset_eval": config.exp.get("use_whole_dataset_eval", False),
        "use_embedding_cache": config.exp.get("use_embedding_cache", False),
        "cache_dir": config.exp.get("cache_dir", "cache"),
        "whole_dataset_eval_interval": config.exp.get("whole_dataset_eval_interval", 1),
    }

    train_genomics_clip(clip_model, train_dataset, val_dataset, train_config)


def train_clip_distributed(config: DictConfig, rank: int, world_size: int, device: str):
    """Distributed training function"""
    if is_main_process(rank):
        print(f"Starting distributed CLIP training with {world_size} GPUs...")

    dataset = get_components(config)

    # Split dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    if is_main_process(rank):
        print(f"Training set size: {train_size}, Validation set size: {val_size}")

    # Create model with parameters from config
    clip_model = GenomicsCLIP(
        cell_vocab_size=config.exp.cell_vocab_size,
        max_cell_tokens=config.exp.max_cell_tokens,
        cell_embed_dim=config.exp.cell_embed_dim,
        cell_transformer_heads=config.exp.cell_transformer_heads,
        cell_transformer_layers=config.exp.cell_transformer_layers,
        text_model_name=config.exp.text_model_name,
        max_text_tokens=config.exp.max_text_tokens,
        text_proj_dim=config.exp.text_proj_dim,
        projection_dim=config.exp.projection_dim,
        dropout=config.exp.dropout,
        device=device,
    )

    clip_model.to(device)

    # Wrap model with DDP
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    clip_model = DDP(clip_model, device_ids=[local_rank])

    # Extract training config parameters
    train_config = {
        "batch_size": config.exp.batch_size,
        "epochs": config.exp.epochs,
        "lr": config.exp.lr,
        "min_lr": config.exp.min_lr,
        "weight_decay": config.exp.weight_decay,
        "num_workers": config.exp.num_workers,
        "model_save_path": config.exp.model_save_path,
        "use_wandb": config.exp.use_wandb
        and is_main_process(rank),  # Only log from main process
        "log_accuracy": config.exp.log_accuracy,
        "wandb_project": config.wandb.project,
        "wandb_entity": config.wandb.entity,
        "log_dir": config.exp.log_dir,
        "use_whole_dataset_eval": config.exp.get("use_whole_dataset_eval", False),
        "use_embedding_cache": config.exp.get("use_embedding_cache", False),
        "cache_dir": config.exp.get("cache_dir", "cache"),
        "whole_dataset_eval_interval": config.exp.get("whole_dataset_eval_interval", 1),
        "rank": rank,
        "world_size": world_size,
    }

    train_genomics_clip(clip_model, train_dataset, val_dataset, train_config)


def train_genomics_clip(
    model: GenomicsCLIP,
    train_dataset,
    val_dataset,
    config: dict,
):
    # Check if we're in distributed training
    is_distributed = hasattr(model, "module")
    rank = config.get("rank", 0)
    world_size = config.get("world_size", 1)

    if config.get("use_wandb", False) and is_main_process(rank):
        wandb_config = {
            "project": config.get("wandb_project", ""),
            "entity": config.get("wandb_entity", ""),
            "config": config,
        }
        # Only include non-empty values
        wandb_config = {k: v for k, v in wandb_config.items() if v}
        wandb.init(**wandb_config)

    # Create distributed samplers if in distributed mode
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = (
            DistributedSampler(
                val_dataset, num_replicas=world_size, rank=rank, shuffle=False
            )
            if val_dataset is not None
            else None
        )
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=(train_sampler is None),  # Don't shuffle if using DistributedSampler
        sampler=train_sampler,
        collate_fn=collate_with_padding,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )

    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            sampler=val_sampler,
            collate_fn=collate_with_padding,
            num_workers=config.get("num_workers", 4),
        )
        if val_dataset is not None
        else None
    )

    # Optimizer and scheduler
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.01),
    )

    # Calculate total steps based on epochs
    total_steps = config["epochs"] * len(train_loader)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=config.get("min_lr", 1e-6)
    )

    # Loss function (symmetric contrastive)
    def clip_loss(logits_per_cell, logits_per_text):
        device = logits_per_cell.device
        labels = torch.arange(logits_per_cell.size(0), device=device)
        cell_loss = F.cross_entropy(logits_per_cell, labels)
        text_loss = F.cross_entropy(logits_per_text, labels)
        return (cell_loss + text_loss) / 2

    best_val_loss = float("inf")
    global_step = 0

    # Get the actual model (unwrap DDP if needed)
    actual_model = model.module if is_distributed else model

    # Training loop
    for epoch in range(config["epochs"]):
        # Set epoch for distributed sampler
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0
        train_accuracy = []

        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}", disable=not is_main_process(rank)
        )
        for batch in pbar:
            optimizer.zero_grad()
            logits_per_cell, logits_per_text = model(batch)
            loss = clip_loss(logits_per_cell, logits_per_text)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Increment global step
            global_step += 1

            # Logging
            train_loss += loss.item()
            if is_main_process(rank):
                pbar.set_postfix({"loss": loss.item(), "step": global_step})

            if config.get("use_wandb", False) and is_main_process(rank):
                logit_scale = actual_model.logit_scale.exp().item()
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/temp": logit_scale,
                        "train/step": global_step,
                    },
                    commit=False,
                )

            if config.get("log_accuracy", False):
                train_accuracy.append(actual_model.accuracy_paired_batch(batch))

            # Save model checkpoint every 100 steps (only on main process)
            if global_step % 100 == 0 and is_main_process(rank):
                avg_train_loss = train_loss / (global_step % len(train_loader) or 1)
                avg_train_accuracy = 0.0
                if config.get("log_accuracy", False):
                    avg_train_accuracy = np.mean(train_accuracy)

                print(
                    f"Step {global_step}: Train Loss = {avg_train_loss:.4f}, Train Accuracy = {avg_train_accuracy:.4f}"
                )

                # Log to wandb
                if config.get("use_wandb", False):
                    wandb.log(
                        {
                            "train/step_loss": avg_train_loss,
                            "train/step_accuracy": avg_train_accuracy,
                            "train/step": global_step,
                        }
                    )

                # Save model checkpoint
                save_path = config.get("model_save_path", "genomics_clip.pt")
                if "log_dir" in config and config["log_dir"]:
                    save_path = os.path.join(config["log_dir"], save_path)

                # Ensure directory exists
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

                # Save state dict of actual model (unwrap DDP)
                torch.save(
                    {
                        "step": global_step,
                        "model_state_dict": actual_model.state_dict(),
                        "train_loss": avg_train_loss,
                    },
                    "model-weights",
                )

                # Create and log wandb artifact
                if config.get("use_wandb", False):
                    art = wandb.Artifact(
                        name="model-weights",
                        type="model",
                        description=f"Model checkpoint at step {global_step} with training loss {avg_train_loss:.4f}",
                    )
                    art.add_file("model-weights")
                    wandb.log_artifact(art)

        # Synchronize all processes before validation
        if is_distributed:
            dist.barrier()

        # Reset metrics for next epoch
        train_loss = 0.0
        train_accuracy = []

        # Validation phase at the end of each epoch
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_accuracy = []

            # Standard batch-wise validation
            with torch.no_grad():
                for val_batch in val_loader:
                    logits_per_cell, logits_per_text = model(val_batch)
                    val_loss += clip_loss(logits_per_cell, logits_per_text).item()
                    if config.get("log_accuracy", False):
                        val_accuracy.append(
                            actual_model.accuracy_paired_batch(val_batch)
                        )

            # Reduce validation metrics across all processes
            if is_distributed:
                val_loss_tensor = torch.tensor(val_loss, device=f"cuda:{rank}")
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                val_loss = val_loss_tensor.item() / world_size

                if config.get("log_accuracy", False) and val_accuracy:
                    val_acc_tensor = torch.tensor(
                        np.mean(val_accuracy), device=f"cuda:{rank}"
                    )
                    dist.all_reduce(val_acc_tensor, op=dist.ReduceOp.SUM)
                    avg_val_accuracy = val_acc_tensor.item() / world_size
                else:
                    avg_val_accuracy = 0.0
            else:
                avg_val_accuracy = np.mean(val_accuracy) if val_accuracy else 0.0

            avg_val_loss = val_loss / len(val_loader)

            if is_main_process(rank):
                print(f"Epoch {epoch + 1}: Val Loss = {avg_val_loss:.4f}")
                if config.get("log_accuracy", False):
                    print(f"Epoch {epoch + 1}: Val Accuracy = {avg_val_accuracy:.4f}")

            # Whole dataset evaluation (if enabled) - all processes participate
            whole_dataset_metrics = {}
            if (
                config.get("use_whole_dataset_eval", False)
                and (epoch + 1) % config.get("whole_dataset_eval_interval", 1) == 0
            ):
                if is_main_process(rank):
                    print("Performing whole dataset evaluation...")

                # All processes participate in distributed evaluation
                whole_dataset_metrics = actual_model.evaluate_whole_dataset(
                    val_loader,
                    use_cache=config.get("use_embedding_cache", False),
                    cache_dir=config.get("cache_dir", "cache"),
                    distributed_strategy=config.get(
                        "distributed_eval_strategy", "gather"
                    ),
                )

                # Only main process prints results (metrics may be empty on other processes)
                if is_main_process(rank) and whole_dataset_metrics:
                    print("Whole dataset evaluation results:")
                    for metric_name, metric_value in whole_dataset_metrics.items():
                        print(f"  {metric_name}: {metric_value:.4f}")

            # Log validation metrics to wandb (only main process)
            if config.get("use_wandb", False) and is_main_process(rank):
                log_dict = {
                    "val/epoch_loss": avg_val_loss,
                    "val/epoch_accuracy": avg_val_accuracy
                    if config.get("log_accuracy", False)
                    else None,
                    "epoch": epoch + 1,
                }

                # Add whole dataset metrics if available (only on main process)
                if whole_dataset_metrics:
                    for metric_name, metric_value in whole_dataset_metrics.items():
                        log_dict[f"val/whole_dataset_{metric_name}"] = metric_value

                wandb.log(log_dict)

            # Save model checkpoint with validation metrics (only main process)
            if is_main_process(rank):
                save_path = config.get("model_save_path", "genomics_clip.pt")
                if "log_dir" in config and config["log_dir"]:
                    save_path = os.path.join(config["log_dir"], save_path)

                checkpoint_data = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "model_state_dict": actual_model.state_dict(),
                    "val_loss": avg_val_loss,
                }

                # Add whole dataset metrics to checkpoint
                if whole_dataset_metrics:
                    checkpoint_data["whole_dataset_metrics"] = whole_dataset_metrics

                torch.save(checkpoint_data, "model-weights")

                # Create and log wandb artifact for validation checkpoint
                if config.get("use_wandb", False):
                    description = f"Model checkpoint at epoch {epoch + 1} with validation loss {avg_val_loss:.4f}"
                    if whole_dataset_metrics:
                        avg_acc = whole_dataset_metrics.get("average_accuracy", 0)
                        description += f" and whole dataset accuracy {avg_acc:.4f}"

                    art = wandb.Artifact(
                        name="model-weights",
                        type="model",
                        description=description,
                    )
                    art.add_file("model-weights")
                    wandb.log_artifact(art)

        # Set model back to training mode
        model.train()

    if config.get("use_wandb", False) and is_main_process(rank):
        wandb.finish()


def collate_with_padding(batch):
    return_dict = {}
    batch_keys = batch[0].keys()

    for key in batch_keys:
        return_dict[key] = torch.nn.utils.rnn.pad_sequence(
            [sample[key] for sample in batch],
            batch_first=True,
            padding_value=0,
        )

    return return_dict
