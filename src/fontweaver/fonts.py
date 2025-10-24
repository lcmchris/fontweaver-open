from __future__ import annotations
from dataclasses import dataclass
import logging
from fontweaver.config import FontweaverConfig, base_config as cfg

from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.boundsPen import ControlBoundsPen
from fontTools.pens.freetypePen import FreeTypePen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

from fontTools.misc.transform import Scale, Offset
import torch
from pathlib import Path
from math import floor

from torch.utils.data import Dataset
from io import BytesIO


class TooManyTokens(BaseException):
    pass


GlyphPaths = list[tuple[str, tuple[tuple[int, int], ...]]]
NormalizedCoords = list[str | float]


class NotSupportedYet(BaseException):
    pass


class FontGlyphMissing(BaseException):
    pass


class MissingBoundary(BaseException):
    pass


class MissingDim(BaseException):
    pass


@dataclass
class Font:
    glyph_paths: dict[str, GlyphPaths]


class Fonts:
    def __init__(
        self,
        path: Path | None = None,
        glyph_set: dict[str, GlyphPaths] | None = None,
    ) -> None:
        self.trg_glyphs = cfg.glyphs

        if path is None and glyph_set is None or path and glyph_set:
            raise ValueError("Only one path or glyph_set")

        if path:
            font = TTFont(path)
            self.glyph_set = font.getGlyphSet()

        elif glyph_set:
            self.glyph_set = glyph_set
        else:
            raise ValueError("Only one path or glyph_set")

        self.font_size = cfg.font_size

        self.glyph_pens: dict[str, RecordingPen] = {}
        self.glyph_paths: Font

    def process_check(
        self,
    ) -> Fonts:
        """Checks for processing glyphs."""
        if not set(self.trg_glyphs).issubset(self.glyph_set.keys()):
            raise FontGlyphMissing(f"Missing glyph in font")
        return self

    def record(self, glyfTable=None, process_check=False) -> Fonts:
        """Extract glyph path data using the pen API."""
        for trg_glyph in self.trg_glyphs:
            pen = RecordingPen()
            try:
                self.glyph_set[trg_glyph].draw(
                    pen
                ) if not glyfTable else self.glyph_set[trg_glyph].draw(
                    pen, glyfTable=glyfTable
                )
                self.glyph_pens[trg_glyph] = pen
            except KeyError:
                self.glyph_pens[trg_glyph] = None
                logging.warning(f"Glyph {trg_glyph} not found in font")
            if process_check:
                sum_points = 2
                for command, points in pen.value:
                    sum_points = sum_points + len(points) * 2 + 1
                if sum_points > cfg.max_glyph_tokens:
                    raise TooManyTokens(
                        f"Glyph {trg_glyph} has too many tokens: {len(pen.value)}"
                    )
        return self

    def _get_transformation(
        self,
    ):
        """Get the bounds of all glyphs and find the max and mins."""
        bounds_pen = ControlBoundsPen(glyphSet=self.glyph_set)
        for trg_glyph in self.trg_glyphs:
            if trg_glyph in [".notdef", ".null", "nonmarkingreturn", "space"]:
                continue
            self.glyph_pens[trg_glyph].replay(bounds_pen)

        if bounds_pen.bounds is None:
            raise MissingBoundary()
        (xMin, yMin, xMax, yMax) = bounds_pen.bounds
        # Scale to fit into scale size

        if xMax - xMin == 0 or yMax - yMin == 0:
            raise MissingDim("Missing dimensions in font")
        x_scale = self.font_size[0] / (xMax - xMin)
        y_scale = self.font_size[1] / (yMax - yMin)

        min_scale = floor(min(x_scale, y_scale) * (10**4)) / 10**4

        assert min_scale != 0

        return Offset(-xMin, -yMin), Scale(min_scale)

    def scale_offset(
        self,
    ):
        offset, scale = self._get_transformation()
        for trg_glyph in self.trg_glyphs:
            recording_pen = RecordingPen()

            scale_pen: TransformPen = TransformPen(
                outPen=recording_pen, transformation=scale
            )
            offset_pen: TransformPen = TransformPen(
                outPen=scale_pen, transformation=offset
            )

            self.glyph_pens[trg_glyph].replay(offset_pen)
            self.glyph_pens[trg_glyph] = recording_pen
        return self

    def draw(
        self,
        name: Path,
        suffix: str = "",
    ) -> Fonts:
        for trg_glyph in self.trg_glyphs:
            free_pen = FreeTypePen(glyphSet=self.glyph_set)
            self.glyph_pens[trg_glyph].replay(free_pen)
            img = free_pen.image()
            img.save(f"{name}_{trg_glyph}_{suffix}.png")
        return self

    def draw_return(
        self,
    ) -> list[BytesIO]:
        binary_ios: list[BytesIO] = []
        for trg_glyph in self.trg_glyphs:
            if len(self.glyph_pens[trg_glyph].value) == 0:
                continue
            binary_io = BytesIO()
            free_pen = FreeTypePen(glyphSet=self.glyph_set)
            self.glyph_pens[trg_glyph].replay(free_pen)
            img = free_pen.image()
            img.save(binary_io, format="png")
            binary_ios.append(binary_io)
        return binary_ios

    def paths(self):
        glyph_paths: dict[str, tuple] = {}
        for trg_glyph in self.trg_glyphs:
            glyph_paths[trg_glyph] = self.glyph_pens[trg_glyph].value
        return Font(glyph_paths)


class FontCodec:
    """Encodes font glyphs from token sequences.
    Each glyph command and coordinate is mapped to a unique integer token.
    """

    def __init__(
        self,
    ) -> None:
        self.glyphs = cfg.glyphs
        self.max_font_tokens = cfg.max_font_tokens

        self.coord_len = cfg.font_size[0] + cfg.font_size[1]

        self.system_tokens = cfg.glyphs + [
            "moveTo",
            "lineTo",
            "qCurveTo",
            "curveTo",
            "closePath",
            "<EOG>",
            "<EOS>",
            "<PAD>",
        ]

        self.mapping: dict[str, int] = {
            token: self.coord_len + i for i, token in enumerate(self.system_tokens)
        }

        self.vocab_size = self.coord_len + len(self.mapping)
        self.reverse_mapping: dict[int | float, str] = {
            value: key for key, value in self.mapping.items()
        }
        self.pad_token = self.mapping["<PAD>"]
        self.eos_token = self.mapping["<EOS>"]
        self.eog_token = self.mapping["<EOG>"]

        self.command_mask = torch.tensor(
            [self.is_command(i) for i in range(self.vocab_size)], device=cfg.device
        )
        self.coord_mask = torch.tensor(
            [not self.is_command(i) for i in range(self.vocab_size)], device=cfg.device
        )
        self.coord_x_mask = torch.tensor(
            [i < cfg.font_size[0] for i in range(self.vocab_size)], device=cfg.device
        )
        self.coord_y_mask = torch.tensor(
            [
                cfg.font_size[0] <= i < cfg.font_size[0] + cfg.font_size[1]
                for i in range(self.vocab_size)
            ],
            device=cfg.device,
        )

    def encode_font(
        self,
        font: Font,
    ) -> dict[str, torch.Tensor]:
        # fill with padding tokens.

        def next_idx():
            nonlocal idx
            result = idx
            idx += 1
            return result

        def encode_point(point: tuple[int, int]) -> tuple[int, int]:
            x, y = point
            assert 0 <= x, f"x={x} is out of bounds"
            assert 0 <= y, f"y={y} is out of bounds"
            return int(x), int(y) + cfg.font_size[0]

        def encode_command(key: str) -> int:
            return self.mapping[key]

        font_tokens_dict = {}

        for glyph_letter, glyph_paths in font.glyph_paths.items():
            idx = 0
            # initialize with padding tokens
            tokens = torch.fill(
                torch.zeros(cfg.max_glyph_tokens, dtype=torch.long, device=cfg.device),
                self.pad_token,
            )
            # start of the glyph
            tokens[next_idx()] = self.mapping[glyph_letter]
            for command, points in glyph_paths:
                if command in ["addComponent", "addVarComponent"]:
                    raise NotSupportedYet("Not supported commands yet.")

                tokens[next_idx()] = encode_command(command)

                for point in points:
                    if point is None:
                        continue
                    x, y = encode_point(point)
                    tokens[next_idx()] = x
                    tokens[next_idx()] = y
            tokens[next_idx()] = self.mapping["<EOG>"]

            font_tokens_dict[glyph_letter] = tokens

        return font_tokens_dict

    def is_command(self, token: int) -> bool:
        return token >= self.coord_len

    def split_font_by_glyphs(
        self,
        font_tokens_dict: dict[str, torch.Tensor],
        num_combined_glyphs: int = 2,
    ) -> list[torch.Tensor]:
        """Splits font into sequentially group of glyphs."""
        font_tokens_dict_num = {
            idx: path for idx, (_, path) in enumerate(font_tokens_dict.items())
        }

        combos = []
        num_glyph = len(font_tokens_dict.values())
        for start in range(num_glyph):  ## a-b ... Y-Z and Z-a
            tensors_to_combine = []
            for i in range(num_combined_glyphs):
                tensor_idx = (start + i) % num_glyph
                tensors_to_combine.append(font_tokens_dict_num[tensor_idx])
            combos.append(torch.cat(tensors_to_combine))

        return combos


@dataclass
class FontDatasetEntry:
    font_tokens: torch.Tensor
    text_embeddings: torch.Tensor
    text_attn_mask_ids: torch.Tensor


class FontDataset(Dataset):
    def __init__(self, ds_path):
        self.data: list[FontDatasetEntry] = []
        self.font_codec = FontCodec()
        for idx, ds_file in enumerate(Path(ds_path).iterdir()):
            if ds_file.suffix == ".ds":
                self.data += torch.load(ds_file, weights_only=False)

        self.data = [d for d in self.data if d is not None]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data[idx]

        return {
            "font_tokens": row.font_tokens,
            "text_embeddings": row.text_embeddings,
            "text_attn_mask_ids": row.text_attn_mask_ids,
        }
