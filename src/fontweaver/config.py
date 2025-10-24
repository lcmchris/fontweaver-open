from dataclasses import dataclass

import torch
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.absolute()


@dataclass
class FontweaverConfig:
    # Preprocessing parameters
    experiment_name: str
    experiment_desc: str
    # raw_dataset: str
    # desc_path: str
    dataset_path: str

    max_text_tokens: int
    max_glyph_tokens: int
    glyphs: list[str]
    num_combined_glyphs: int

    # Model parameters
    val_perc = 0.04
    d_model: int
    nhead: int
    num_layers: int

    max_epochs: int
    learning_rate: float
    batch_size: int
    font_size: tuple[int, int]
    bert_model: str

    profiler: str
    checkpoint_path: str = None

    def __post_init__(self):
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.precision = "bf16-mixed" if self.device == "cuda" else 32
        self.max_font_tokens = self.max_glyph_tokens * len(self.glyphs)
        self.max_seq_len = self.max_text_tokens + self.max_font_tokens


base_config = FontweaverConfig(
    experiment_name="fonts-and-ai",
    experiment_desc="Fonts and Ai.",
    dataset_path="2025-10-23-fonts-and-ai-demo",
    glyphs=["a", "b", "c"],  # List of glyphs to include in training
    num_combined_glyphs=2,  # Number of glyph sequences to combine as a sample
    d_model=512,
    nhead=4,
    num_layers=8,
    max_text_tokens=4,  # Number of text tokens to included
    max_glyph_tokens=256,
    learning_rate=0.001,
    batch_size=16,
    font_size=(100, 100),
    bert_model="google/bert_uncased_L-12_H-512_A-8",
    max_epochs=15,
    profiler="advanced",
    # checkpoint_path=str( ## for resuming training from a checkpoint
    #     ROOT_DIR
    #     / "logs"
    #     / "final-experiment-2"
    #     / "version_4"
    #     / "checkpoints"
    #     / "last.ckpt"
    # ),
)
