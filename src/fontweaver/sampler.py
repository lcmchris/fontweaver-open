from __future__ import annotations
import random

from abc import abstractmethod
import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from fontweaver.config import base_config as cfg, ROOT_DIR
from fontweaver.fonts import Fonts, Font, GlyphPaths
from fontweaver.model import FontweaverModel, TextEmbedder
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen, Glyph

from pathlib import Path
from abc import ABC
from io import BytesIO
from typing import Literal
import logging

from pydantic import BaseModel
from datetime import datetime
import typer

Strategy = Literal["greedy", "multinomial"]


class SamplingCallback(pl.Callback):
    def __init__(
        self,
        sample_every_epoch: int,
    ):
        super().__init__()
        self.device = cfg.device
        self.sample_every_epoch = sample_every_epoch

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if (trainer.current_epoch + 1) % self.sample_every_epoch == 0:
            model: FontweaverModel = pl_module
            model.eval()
            try:
                sampler = FontweaverSampler(
                    model=model,
                    outputer=OutputPNG(),
                    exporter=ExportLocal(
                        out_folder=Path(trainer.log_dir) / "samples",
                    ),
                    temperature=0.8,
                    strategy="multinomial",
                )
                text = "Clean"
                sampler.sample_main(
                    font_metadata=FontMetadata(
                        name=text + str(trainer.current_epoch),
                        prompt=text,
                    ),
                )
            except Exception as e:
                print("failed to sample", e)
            model.train()


class OutputBase(ABC):
    def __init__(
        self,
    ):
        self.suffix: str
        return

    @abstractmethod
    def generate(self, font: Font) -> list[BytesIO]:
        pass

    def record_glyph_paths(self, font: Font) -> dict[str, Glyph]:
        glyphs = {".notdef": TTGlyphPen(glyphSet={}).glyph()}
        for glyph_name, path_data in font.glyph_paths.items():
            if path_data is None:
                continue
            pen = TTGlyphPen(glyphSet={})
            for idx, (command, data) in enumerate(path_data):
                if command == "moveTo":
                    pen.moveTo(*data)
                elif command == "lineTo":
                    pen.lineTo(*data)
                elif command == "qCurveTo":
                    pen.qCurveTo(*data)
                elif command == "curveTo":
                    pen.curveTo(*data)
                elif command == "closePath":
                    pen.closePath()
                else:
                    raise Exception(f"Unknown command: {command}")

                if idx == len(path_data) - 1 and command != "closePath":
                    # If the last command is not closePath, we need to close it
                    pen.closePath()

            glyphs[glyph_name] = pen.glyph()

        return glyphs


class OutputTTF(OutputBase):
    def __init__(
        self,
    ):
        super().__init__()
        self.suffix = ".ttf"

    def drawDummyGlyph(self, pen: TTGlyphPen):
        pen.moveTo((100, 100))
        pen.lineTo((100, 1000))
        pen.qCurveTo((200, 900), (400, 900), (500, 1000))
        pen.lineTo((500, 100))
        pen.closePath()

    def generate(self, font: Font) -> list[BytesIO]:
        fb = FontBuilder(unitsPerEm=100, isTTF=True)
        cmap = {}
        glyph_order = []
        dummy_glyphs = [".notdef", "space", ".null"]
        for glyph in dummy_glyphs + list(font.glyph_paths.keys()):
            if glyph not in dummy_glyphs:
                cmap[ord(glyph)] = glyph
            glyph_order.append(glyph)

        pen = TTGlyphPen(None)
        self.drawDummyGlyph(pen)
        dummy_glyph = pen.glyph()
        dummy_glyphs_dict = {
            ".notdef": dummy_glyph,
            "space": dummy_glyph,
            ".null": dummy_glyph,
        }

        fb.setupGlyphOrder(glyph_order)
        fb.setupCharacterMap(cmap)

        ttglyph_map = self.record_glyph_paths(font=font)
        ttglyph_map = {**dummy_glyphs_dict, **ttglyph_map}
        fb.setupGlyf(ttglyph_map)

        h_metrics = {}
        v_metrics = {}
        glyph_table = fb.font["glyf"]

        for glyph, _ in ttglyph_map.items():
            width_boundary = cfg.font_size[0]
            glyph_width = glyph_table[glyph].xMax - glyph_table[glyph].xMin
            h_metrics[glyph] = (cfg.font_size[0], 0)
            v_metrics[glyph] = (cfg.font_size[1], 0)

        fb.setupHorizontalMetrics(h_metrics)
        fb.setupHorizontalHeader(ascent=0, descent=0)

        fb.setupVerticalMetrics(v_metrics)
        fb.setupVerticalHeader(ascent=0, descent=0)

        familyName = "Fontweaver"
        styleName = "FontweaverGenerated"
        version = "0.1"

        nameStrings = dict(
            familyName=dict(
                en=familyName,
            ),
            styleName=dict(
                en=styleName,
            ),
            uniqueFontIdentifier="fontBuilder: " + familyName + "." + styleName,
            fullName=familyName + "-" + styleName,
            psName=familyName + "-" + styleName,
            version="Version " + version,
        )
        fb.setupNameTable(nameStrings)

        fb.setupOS2()
        fb.setupPost()

        binary_bytes = BytesIO()
        fb.save(binary_bytes)
        return [binary_bytes]


class OutputPNG(OutputBase):
    def __init__(
        self,
    ):
        super().__init__()
        self.suffix = ".png"

    def generate(self, font: Font) -> list[BytesIO]:
        ttfglyphs = self.record_glyph_paths(font=font)
        fonts = Fonts(path=None, glyph_set=ttfglyphs)
        binary_images = fonts.record(glyfTable=ttfglyphs).draw_return()
        return binary_images


class FontMetadata(BaseModel):
    name: str
    prompt: str
    bucket_id: str | None = None
    object_path: str | None = None


class ExportBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def save(
        self, name: Path, bytesIOs: list[BytesIO], font_metadata: FontMetadata, suffix
    ) -> Path:
        pass


class ExportLocal(ExportBase):
    def __init__(self, out_folder: Path):
        super().__init__()
        self.out_folder = out_folder

    def save(
        self, name: Path, bytesIOs: list[BytesIO], font_metadata: FontMetadata, suffix
    ) -> Path:
        if not self.out_folder.exists():
            self.out_folder.mkdir(parents=True)

        for idx, bytesIO in enumerate(bytesIOs):
            with open(self.out_folder / f"{name}_{idx}_{suffix}", "wb") as f:
                f.write(bytesIO.getbuffer())
                print(f"Saved to {self.out_folder}/{name}_{idx}_{suffix}")
        return self.out_folder


class FontweaverSampler:
    def __init__(
        self,
        model: FontweaverModel,
        outputer: OutputBase,
        exporter: ExportBase,
        temperature: float,
        strategy: Strategy,
        glyphs: list[str] | None = None,
    ):
        self.model = model
        self.outputer = outputer
        if glyphs is None:
            glyphs = cfg.glyphs
        self.glyphs = glyphs
        self.font_codec = model.font_codec
        self.max_glyph_tokens = cfg.max_glyph_tokens
        self.max_font_tokens = cfg.max_font_tokens
        self.exporter = exporter
        self.device = cfg.device
        # do not consume GPU memory for the text module
        self.text_embedder = TextEmbedder()

        self.temperature = temperature
        self.strategy = strategy

        self.text_embeddings: torch.Tensor
        self.font_tokens: torch.Tensor
        self.glyph_paths: dict[str, GlyphPaths] = {glyph: [] for glyph in self.glyphs}
        self.idx = 0

    def sample_next_token(
        self,
        logit: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        max_index: torch.Tensor
        logit /= self.temperature
        max_index: torch.Tensor
        if mask is not None:
            logit = torch.where(mask, logit, torch.full_like(logit, -float("Inf")))
        if self.strategy == "multinomial":
            probabilities = torch.softmax(logit, dim=0)
            max_index = torch.multinomial(probabilities, num_samples=1)
        elif self.strategy == "greedy":
            probabilities = torch.softmax(logit, dim=0)
            max_index = torch.argmax(probabilities, dim=0)
        elif self.strategy == "topknuc":
            # Top-K and/or Nucleus Filtering
            top_k = 10
            top_p = 0.9
            logit = top_k_top_p_filtering(logit, top_k=top_k, top_p=top_p)

            probabilities = torch.softmax(logit, dim=0)
            max_index = torch.multinomial(probabilities, num_samples=1)
        else:
            raise ValueError(f"Unknown type {self.strategy}")

        return max_index.item()

    def update_fonts_tokens(self, next_token: int) -> None:
        self.font_tokens[self.idx] = next_token
        self.idx += 1
        if self.idx >= self.max_glyph_tokens:
            raise ValueError("Font too long")
        self.font_tokens[self.idx] = (
            self.font_codec.eog_token
        )  ## appending some random token
        # self.font_tokens[self.idx + 1] = self.font_codec.eos_token

    def sample_point(
        self,
    ) -> tuple[int, int]:
        predicted = self.prediction_proba()
        x = self.sample_next_token(predicted, mask=self.font_codec.coord_mask)
        self.update_fonts_tokens(x)
        predicted = self.prediction_proba()
        y = self.sample_next_token(predicted, mask=self.font_codec.coord_y_mask)
        self.update_fonts_tokens(y)
        y = y - cfg.font_size[0]
        return x, y

    def sample_y(
        self,
    ) -> int:
        predicted = self.prediction_proba()
        y = self.sample_next_token(predicted, mask=self.font_codec.coord_y_mask)
        self.update_fonts_tokens(y)
        y = y - cfg.font_size[0]
        return y

    def sample_command(
        self,
    ):
        predicted = self.prediction_proba()
        command = self.sample_next_token(predicted, mask=self.font_codec.command_mask)
        if command == self.font_codec.eos_token or command == self.font_codec.eog_token:
            self.idx += 1
            return command
        self.update_fonts_tokens(command)
        return command

    def sample_any(
        self,
    ):
        predicted = self.prediction_proba()
        token = self.sample_next_token(predicted)
        self.update_fonts_tokens(token)
        return token

    def prediction_proba(
        self,
    ) -> torch.Tensor:
        font_out = self.model(
            text_embeddings=self.text_embeddings,
            text_attn_mask_ids=self.text_attn_mask_ids,
            font_tokens=torch.stack(
                [self.font_tokens],
            ),
        )
        last_predicted = font_out[0][self.idx]  # B, S, Prob
        return last_predicted

    def sample_font(
        self,
    ) -> Font:
        """Recursively samples the font from the model"""
        self.current_glyph_path: GlyphPaths = []

        self.glyph_idx = 0
        self.glyph_letter = self.glyphs[self.glyph_idx]
        self.idx = 0

        self.font_tokens = torch.full(
            (self.max_glyph_tokens,),
            self.font_codec.pad_token,
            device=cfg.device,
        )
        self.update_fonts_tokens(
            self.font_codec.mapping[self.glyph_letter]
        )  # first token

        arg = None
        while True:
            logging.info(f"Sampling {self.glyph_letter}")
            command_token = self.sample_command() if arg is None else arg
            arg = None
            command = self.font_codec.reverse_mapping[command_token]
            if self.idx >= self.max_font_tokens - 1:
                return Font(glyph_paths=self.glyph_paths)
            if len(self.glyph_paths[self.glyph_letter]) > self.max_glyph_tokens:
                raise Exception("Glyph too long")

            if self.idx >= self.max_glyph_tokens - 20:
                command = "<EOG>"

            if command == "<EOS>":
                return Font(glyph_paths=self.glyph_paths)
            elif command == "<EOG>":
                self.glyph_paths[self.glyph_letter] = self.current_glyph_path
                self.current_glyph_path = []

                self.glyph_idx += 1
                if self.glyph_idx == len(self.glyphs):
                    return Font(glyph_paths=self.glyph_paths)

                self.glyph_letter = self.glyphs[self.glyph_idx]

                eog_idx = (
                    self.font_tokens == self.font_codec.mapping["<EOG>"]
                ).nonzero(as_tuple=True)
                if len(eog_idx[0]) >= 2:
                    start = eog_idx[0][-2].item()
                    end = eog_idx[0][-1].item()

                    replace = torch.full(
                        (self.max_glyph_tokens,),
                        self.font_codec.pad_token,
                        device=cfg.device,
                    )
                    replace[: end - start] = self.font_tokens[start + 1 : end + 1]
                    self.font_tokens = replace
                    self.idx = end - start
                self.update_fonts_tokens(self.font_codec.mapping[self.glyph_letter])
            elif command == "<PAD>":
                # skip PAD tokens
                continue
            elif command in self.glyphs:
                arg = self.font_codec.mapping["<EOG>"]
                # self.update_fonts_tokens(self.font_codec.mapping["<EOG>"])
                # self.glyph_paths[self.glyph_letter] = self.current_glyph_path

                # self.current_glyph_path = []
            elif command == "closePath":
                self.current_glyph_path.append((command, tuple()))
            elif command == "moveTo":
                point_x, point_y = self.sample_point()
                data_moveTo = ((point_x, point_y),)
                self.current_glyph_path.append((command, data_moveTo))
            elif command == "lineTo":
                point_x, point_y = self.sample_point()
                data_lineTo = ((point_x, point_y),)
                self.current_glyph_path.append((command, data_lineTo))
            elif command == "qCurveTo":
                control_points = []
                count = 0  # minimum 2 control points
                while True:
                    arg = self.sample_any()
                    if arg in self.font_codec.reverse_mapping and count >= 2:
                        break
                    point_x = arg
                    point_y = self.sample_y()
                    control_points.append((point_x, point_y))
                    count += 1
                self.current_glyph_path.append((command, tuple(control_points)))
            elif command == "curveTo":
                control_points = []
                for _ in range(3):  # minimum 3 control points
                    point_x, point_y = self.sample_point()
                    control_points.append((point_x, point_y))
                self.current_glyph_path.append((command, tuple(control_points)))
            else:
                raise Exception(f"Unknown command {command}")

    def sample_main(
        self,
        font_metadata: FontMetadata,
        output_path: Path | None = None,
    ) -> Path:
        print(f"sampling {font_metadata.prompt}")
        text_embeddings, text_attn_mask_ids = self.text_embedder.get_text_emeddings(
            font_metadata.prompt
        )
        self.text_embeddings = text_embeddings.to(self.device)
        self.text_attn_mask_ids = text_attn_mask_ids.bool().to(self.device)
        out_font: Font = self.sample_font()

        if type(self.outputer) is OutputPNG:
            suffix = ".png"
        elif type(self.outputer) is OutputTTF:
            suffix = ".ttf"
        else:
            raise Exception(f"Unknown output type {type(self.outputer)}")

        name: Path = (
            Path(
                f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{font_metadata.name.replace(' ', '_')}"
            )
            if output_path is None
            else output_path
        )

        font_bytes = self.outputer.generate(out_font)

        save_path = self.exporter.save(
            name, font_bytes, font_metadata=font_metadata, suffix=suffix
        )
        return save_path


DEFAULT_SAMPLE_TEXTS = [
    "Elegant",
    "Clean",
    "Authoritative",
    "Dynamic",
]


app = typer.Typer()


@app.command()
def sample_font(text: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info(f"Sampling font for text: {text}")
    output = OutputPNG()
    latest = (ROOT_DIR / "logs" / cfg.experiment_name).iterdir()
    latest_dir = sorted(latest, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    logging.info(f"Latest dir: {latest_dir}")

    model = FontweaverModel.load_from_checkpoint(
        checkpoint_path=latest_dir
        / "checkpoints"
        / "epoch=19-step=3160.ckpt",  # "last.ckpt",
        map_location=cfg.device,
    )
    model.eval()
    with torch.no_grad():
        sampler = FontweaverSampler(
            model=model,
            outputer=output,
            exporter=ExportLocal(out_folder=ROOT_DIR / "samples"),
            temperature=1.0,
            strategy="greedy",
        )

        out = sampler.sample_main(font_metadata=FontMetadata(name=text, prompt=text))
        print(f"Sample save to {out}")
