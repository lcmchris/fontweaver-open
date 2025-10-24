# Open source font model

Training code of a decoder-only model for font generation.
This simple model takes a one-word prompt to generate a font.

```
uv run cli.py sampler sample Clean
```

## Describing a glyph and a font

Font files come in many formats, but each glyph IS essentially comprised on 6 basic commands on a 2d grid. These commands definitions from fontTools are as follows.

- moveTo: Moves the point
- lineTo: Draws line from current point to new point.
- curveTo: Draw a cubic Bézier with an arbitrary number of control points.
- qCurveTo: Draw a whole string of quadratic curve segments. 3
- endPath: End the current sub path, but don’t close it.
- closePath: Close the current sub path.

```
Glyph = moveTo (x,y) >> lineTo (x,y) >> curveTo (x1,y1) (x2,y2) >> ... >> closePath
```

Now a font is not just one glyph, it is a collection of multiple.

```
Font = Glyph 1 >> Glyph 2 >> ... >> Glyph N
```

Both of these lends itself to a next-token based prediction model. Each next token, depends on the last ones (the context).

## Model architecture

Text embedders: Converts text to tokens in ndim space.
Font token embedders: Converts functional font commands to learnable tokens ndim space.
Decoder: Next sequence predictor of ndim space
Logits converter: Takes ndim space back to functional font commands.

The data sequence for each individual font is as follows:

prompt + A + B

prompt + B + C

prompt + C + A

**Tech**: Framework Pytorch lightning and Decoder-Only via x_transformers

### Parsing Dataset

From a collection of `.ttf` files, generate a collection of `.txt` description of fonts using GPT/GEMMA. This is then collected into a csv file.

```sh
uv run cli.py process gemma data/raw/ data/gemma-adjectives
uv run cli.py process collect data/gemma-adjectives data/gemma-adjectives

# _df_desc.csv output csv will look like
# name,adjective
# ScheherazadeNew-Bold.ttf,Opulent
```

Generate a collection of tensors ready for training. This helps reduce cpu usage and it removes in training transformation.

```sh
# class FontDatasetEntry:
#     font_tokens: torch.Tensor
#     text_embeddings: torch.Tensor
#     text_attn_mask_ids: torch.Tensor

uv run cli.py process generate_tensors
```

### Training config and Training

To start training:

```
uv run cli.py train
```

Main config params in `config.py`

- glyphs: default=["a", "b", "c"], List of glyphs to include in training
- num_combined_glyphs: default=2, Number of glyph sequences to combine as a sample
