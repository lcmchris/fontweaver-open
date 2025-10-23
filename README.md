### Open source font model

Training a decoder-only model for font generation.

## Describing a glyph

Font files come in many formats, but each glyph is comprised on 6 basic commands on a 2d grid. The following are command definitions from fontTools, the library I used to parse font files.

moveTo - Moves the point
lineTo - Draws line from current point to new point.
curveTo - Draw a cubic Bézier with an arbitrary number of control points.
qCurveTo - Draw a whole string of quadratic curve segments. 3
endPath - End the current sub path, but don’t close it.
closePath - Close the current sub path.

Glyph = moveTo (x,y) >> lineTo (x,y) >> curveTo (x1,y1) (x2,y2) >> lineTo >> ... >> closePath

## Joining glyphs

A font in is collecting all the
Font = Glyph 1 + Glyph 2 + ... + Glyph N

Both of these lends itself to a next-token based prediction model.

### Parsing Dataset

```sh
# From a collection of `.ttf` files, generate a collection of `.txt` description of fonts using GPT/GEMMA. This is then collected into a csv file.
uv run python cli.py process gemma data/raw/ data/gemma-adjectives
uv run python cli.py process collect data/gemma-adjectives data/gemma-adjectives

output csv:
# name,adjective
# ScheherazadeNew-Bold.ttf,Opulent
# Mitr-Medium.ttf,Approachable
# Stylish-Regular.ttf,Chic
# EncodeSansSemiCondensed-Medium.ttf,Functional

# Generate a collection of tensors ready for training. This helps reduce cpu usage and it removes in training transformation.

# class FontDatasetEntry:
#     font_tokens: torch.Tensor
#     text_embeddings: torch.Tensor
#     text_attn_mask_ids: torch.Tensor

uv run python cli.py process generate_tensors

```

### Quick Start
