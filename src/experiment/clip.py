import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel
from hydra.utils import instantiate
import os
from torch.utils.data import random_split


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
        self.cell_embedding = nn.Embedding(
            cell_vocab_size, cell_embed_dim, device=device
        )
        self.cell_pos_embedding = nn.Parameter(
            torch.randn(1, max_cell_tokens, cell_embed_dim, device=device)
        )

        cell_encoder_layers = TransformerEncoderLayer(
            d_model=cell_embed_dim,
            nhead=cell_transformer_heads,
            dim_feedforward=cell_embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            device=device,
        )
        self.cell_encoder = TransformerEncoder(
            cell_encoder_layers, cell_transformer_layers
        )

        # ============= Text Encoder =============
        self.text_encoder = BertModel.from_pretrained(text_model_name).to(device)

        # Freeze BERT
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # ============= Projection Heads =============
        self.cell_proj = nn.Sequential(
            nn.Linear(cell_embed_dim, projection_dim, device=device),
            nn.GELU(),
            nn.LayerNorm(projection_dim, device=device),
            nn.Linear(projection_dim, projection_dim, device=device),
        )

        self.text_proj = nn.Sequential(
            nn.Linear(
                self.text_encoder.config.hidden_size, text_proj_dim, device=device
            ),
            nn.GELU(),
            nn.LayerNorm(text_proj_dim, device=device),
            nn.Linear(text_proj_dim, projection_dim, device=device),
        )

        # Temperature parameter
        self.logit_scale = nn.Parameter(
            torch.ones([], device=device)
            * torch.log(torch.tensor(1 / 0.07, device=device))
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
        # Encode both modalities
        # cell_tokens = torch.stack(
        #     [self.tokenize_cells(cell) for cell in batch["cell_data"]]
        # )
        # text_tokens, attention_masks = zip(
        #     *[self.tokenize_text(text) for text in batch["text"]]
        # )

        # text_tokens = torch.stack(text_tokens)
        # attention_masks = torch.stack(attention_masks)

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
    print("starting clip training...")
    dataset = get_components(config)

    # Split dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Training set size: {train_size}, Validation set size: {val_size}")

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

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
    }

    train_genomics_clip(clip_model, train_dataset, val_dataset, train_config)


def train_genomics_clip(
    model: GenomicsCLIP,
    train_dataset,
    val_dataset,
    config: dict,
):
    if config.get("use_wandb", False):
        wandb_config = {
            "project": config.get("wandb_project", ""),
            "entity": config.get("wandb_entity", ""),
            "config": config,
        }
        # Only include non-empty values
        wandb_config = {k: v for k, v in wandb_config.items() if v}
        wandb.init(**wandb_config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_with_padding,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )

    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
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
        labels = torch.arange(logits_per_cell.size(0), device=model.device)
        cell_loss = F.cross_entropy(logits_per_cell, labels)
        text_loss = F.cross_entropy(logits_per_text, labels)
        return (cell_loss + text_loss) / 2

    best_val_loss = float("inf")
    global_step = 0
    log_interval = 100  # Log and save every 1000 steps

    # Training loop
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        train_accuracy = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
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
            pbar.set_postfix({"loss": loss.item(), "step": global_step})

            if config.get("use_wandb", False):
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/temp": model.logit_scale.exp().item(),
                        "train/step": global_step,
                    },
                    commit=False,
                )

            if config.get("log_accuracy", False):
                train_accuracy.append(model.accuracy_paired_batch(batch))

            # Save model checkpoint every 100 steps
            if global_step % 100 == 0:
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

                torch.save(
                    {
                        "step": global_step,
                        "model_state_dict": model.state_dict(),
                        # "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": avg_train_loss,
                    },
                    # f"{global_step}_step_train_{save_path}",
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

                # Reset metrics for next interval
                train_loss = 0.0
                train_accuracy = []

        # Validation phase at the end of each epoch
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_accuracy = []
            with torch.no_grad():
                for val_batch in val_loader:
                    logits_per_cell, logits_per_text = model(val_batch)
                    val_loss += clip_loss(logits_per_cell, logits_per_text).item()
                    if config.get("log_accuracy", False):
                        val_accuracy.append(model.accuracy_paired_batch(val_batch))

            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch + 1}: Val Loss = {avg_val_loss:.4f}")

            if config.get("log_accuracy", False):
                avg_val_accuracy = np.mean(val_accuracy)
                print(f"Epoch {epoch + 1}: Val Accuracy = {avg_val_accuracy:.4f}")

            # Log validation metrics to wandb
            if config.get("use_wandb", False):
                wandb.log(
                    {
                        "val/epoch_loss": avg_val_loss,
                        "val/epoch_accuracy": avg_val_accuracy
                        if config.get("log_accuracy", False)
                        else None,
                        "epoch": epoch + 1,
                    }
                )

            # Save model checkpoint with validation metrics
            save_path = config.get("model_save_path", "genomics_clip.pt")
            if "log_dir" in config and config["log_dir"]:
                save_path = os.path.join(config["log_dir"], save_path)

            torch.save(
                {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    # "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                },
                "model-weights",
            )

            # Create and log wandb artifact for validation checkpoint
            if config.get("use_wandb", False):
                art = wandb.Artifact(
                    name="model-weights",
                    type="model",
                    description=f"Model checkpoint at epoch {epoch + 1} with validation loss {avg_val_loss:.4f}",
                )
                art.add_file("model-weights")
                wandb.log_artifact(art)

        # Set model back to training mode
        model.train()

    if config.get("use_wandb", False):
        wandb.finish()


def collate_with_padding(batch):
    return_dict = {}
    batch_keys = batch[0].keys()

    for key in batch_keys:
        return_dict[key] = torch.nn.utils.rnn.pad_sequence(
            [sample[key] for sample in batch],
            batch_first=True,
            padding_side="right",
        )

    return return_dict
