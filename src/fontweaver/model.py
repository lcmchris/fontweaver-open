from typing import Any
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
from transformers import BertModel, BertTokenizer
from fontweaver.fonts import FontCodec, FontDataset
from fontweaver.config import base_config as cfg, ROOT_DIR
import os
from x_transformers import Decoder
from x_transformers.x_transformers import AbsolutePositionalEmbedding
from math import ceil
from torch.optim.lr_scheduler import CosineAnnealingLR

import bitsandbytes as bnb


class FontweaverDataModule(pl.LightningDataModule):
    """This is the data module. It loads the tensors dataset and creates train/val splits."""

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.dataset = FontDataset(ds_path=ROOT_DIR / "data" / cfg.dataset_path)
        dataset_size = len(self.dataset)

        val_size = int(ceil(dataset_size * cfg.val_perc))
        train_size = dataset_size - val_size

        g = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_size, val_size], generator=g
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            pin_memory=True,
            num_workers=int(os.cpu_count()) - 4,
            batch_size=cfg.batch_size,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            num_workers=int(4),
            batch_size=cfg.batch_size,
        )


class FontEmbeddings(nn.Module):
    def __init__(
        self,
    ):
        super(FontEmbeddings, self).__init__()
        self.font_codec = FontCodec()
        self.coord_size = cfg.font_size[0] + cfg.font_size[1]
        self.embedding = nn.Embedding(
            self.font_codec.vocab_size,
            cfg.d_model,
            padding_idx=self.font_codec.pad_token,
        )

        self.coord_pad = 0
        self.coord_embedding_x = nn.Embedding(
            cfg.font_size[0] + 1,
            cfg.d_model,
            padding_idx=self.coord_pad,
        )
        self.coord_embedding_y = nn.Embedding(
            cfg.font_size[1] + 1,
            cfg.d_model,
            padding_idx=self.coord_pad,
        )

        self.coord_len = self.font_codec.coord_len
        self.pos_embedding = AbsolutePositionalEmbedding(
            cfg.d_model, cfg.max_font_tokens
        )

    def font_to_vec(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        points_x = torch.zeros_like(tokens, device=cfg.device)
        points_y = torch.zeros_like(tokens, device=cfg.device)

        x_coord_inds = tokens < cfg.font_size[0]
        y_coord_inds = torch.logical_and(
            tokens >= cfg.font_size[0],
            tokens < cfg.font_size[1] + cfg.font_size[0],
        )

        points_x[x_coord_inds] = tokens[x_coord_inds]
        points_y[y_coord_inds] = tokens[y_coord_inds]

        x_coords = points_x + 1
        y_coords = points_y + 1 - y_coord_inds * cfg.font_size[0]

        return x_coords, y_coords

    def forward(self, font_tokens: torch.Tensor):
        # x: (batch_size, seq_len)

        x_coords, y_coords = self.font_to_vec(font_tokens)

        # Embedding for the token
        token_embedding = self.embedding(font_tokens)  # (batch_size, seq_len, d_model)
        # Final embedding
        coord_embedding_x = self.coord_embedding_x(
            x_coords
        )  # (batch_size, seq_len, d_model)
        coord_embedding_y = self.coord_embedding_y(
            y_coords
        )  # (batch_size, seq_len, d_model)

        pos_embedding = self.pos_embedding(font_tokens)

        embedding = (
            token_embedding + coord_embedding_x + coord_embedding_y + pos_embedding
        )

        return embedding


class FontweaverTransformer(nn.Module):
    def __init__(
        self,
    ):
        super(FontweaverTransformer, self).__init__()
        self.decoder = (
            Decoder(
                dim=cfg.d_model,
                depth=cfg.num_layers,
                heads=cfg.nhead,
                attn_flash=True,
            ).cuda()
            if torch.cuda.is_available()
            else Decoder(
                dim=cfg.d_model,
                depth=cfg.num_layers,
                heads=cfg.nhead,
                attn_flash=False,
            )
        )

    def forward(self, **kwargs):
        return self.decoder(**kwargs)


class TextEmbedder:
    """Uses BERT to tokenize and embed text inputs"""

    def __init__(
        self,
    ):
        self.max_text_tokens = cfg.max_text_tokens
        self.tokenizer = BertTokenizer.from_pretrained(cfg.bert_model)
        self.bert = BertModel.from_pretrained(cfg.bert_model)
        self.device = cfg.device

    def tokenize_batch(self, text_batch: list[str]) -> tuple[Any, Any]:
        encoded_input = self.tokenizer.batch_encode_plus(
            text_batch,
            add_special_tokens=True,
            max_length=self.max_text_tokens,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)
        input_ids = encoded_input["input_ids"]
        attn_mask_ids = encoded_input["attention_mask"]
        return input_ids, attn_mask_ids

    def embed_tokens(self, text_tokens: torch.Tensor):
        text_tokens = text_tokens.to(self.bert.device)
        with torch.no_grad():
            batch_embeddings = self.bert.embeddings(text_tokens)
        return batch_embeddings

    def get_text_emeddings(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        text_tokens, attn_mask_ids = self.tokenize_batch([text])
        text_embeddings = self.embed_tokens(text_tokens)
        return text_embeddings, attn_mask_ids


class FontweaverModel(pl.LightningModule):
    """Main Model with:
    - Text Embedder
    - Font Embeddings
    - Transformer
    - Output layer back to functional commands size"""

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.font_codec = FontCodec()
        self.bert_tokenizer_pad_token_id = 0
        self.seq_len = cfg.max_text_tokens + cfg.max_glyph_tokens
        self.block_size = cfg.batch_size

        self.text_embedder = TextEmbedder()
        self.font_embedder = FontEmbeddings()
        self.transformer = FontweaverTransformer()
        self.to_logits = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, self.font_codec.vocab_size, bias=False),
        )

    def training_step(
        self, batch: dict[str, list[str]], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        step_loss = self.make_a_step(batch)
        self.log_dict(
            {
                "loss": step_loss,
            },
            batch_size=cfg.batch_size,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return {"loss": step_loss}

    def validation_step(
        self, batch: dict[str, list[str]], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        step_loss = self.make_a_step(batch)
        self.log_dict(
            {
                "val_loss": step_loss,
            },
            batch_size=cfg.batch_size,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return {"val_loss": step_loss}

    def configure_optimizers(
        self,
    ) -> tuple[list[torch.optim.Optimizer], list[Any]]:
        """Create a AdamW optimizer and a cosine annealing learning rate scheduler."""
        optimizer = bnb.optim.adamw.AdamW(self.parameters(), lr=cfg.learning_rate)
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = {
            "scheduler": CosineAnnealingLR(
                optimizer,
                T_max=int(total_steps),
                eta_min=cfg.learning_rate / 5,
                last_epoch=-1,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]

    def make_a_step(self, batch) -> torch.Tensor:
        """"""
        font_tokens = batch["font_tokens"]
        text_embeddings = batch["text_embeddings"]
        text_attn_mask_ids = batch["text_attn_mask_ids"]

        font_pred_shifted = self.forward(
            font_tokens=font_tokens,
            text_embeddings=text_embeddings,
            text_attn_mask_ids=text_attn_mask_ids,
        )

        B, L, D = font_pred_shifted.shape

        font_tokens = font_tokens.view(-1)
        font_loss = F.cross_entropy(
            font_pred_shifted.reshape(B * L, D),
            font_tokens,
            ignore_index=self.font_codec.pad_token,
        )

        return font_loss

    def forward(
        self,
        font_tokens: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_attn_mask_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, _ = text_embeddings.shape

        font_embeddings = self.font_embedder(font_tokens)

        # start of sequence token - bump one ahead
        sos_value = 0.42
        sos_token = torch.full(
            (batch_size, 1, cfg.d_model), sos_value, device=self.device
        )
        full_seq = torch.cat(
            [sos_token, text_embeddings, font_embeddings[:, :-1, :]], dim=1
        )

        font_attn_mask_ids = font_tokens != self.font_codec.pad_token

        # For attn_mask Do not attend = True, without the sos token. We do not want to attend to padding tokens.
        full_attn_mask_ids = torch.cat(
            [
                text_attn_mask_ids,
                font_attn_mask_ids,
            ],
            dim=1,
        )

        output = self.transformer(
            x=full_seq,
            mask=full_attn_mask_ids,
        )

        logits = self.to_logits(output)
        font_probabilities = logits[:, cfg.max_text_tokens :, :]
        return font_probabilities

    def on_after_backward(self):
        for name, param in self.named_parameters():
            if param.grad is None:
                print(name)
