import torch
from fontweaver.fonts import (
    Fonts,
    FontCodec,
    FontGlyphMissing,
    MissingBoundary,
    MissingDim,
    TooManyTokens,
    NotSupportedYet,
    FontDatasetEntry,
)
from fontweaver.model import TextEmbedder
import logging
from fontweaver.config import ROOT_DIR, base_config as cfg
import pandas as pd
from multiprocessing import Pool
from pathlib import Path
from pandas import Series
import tqdm
from fontTools.ttLib import TTLibError
import typer
from openai import OpenAI
from dotenv import dotenv_values

import json
import time
from google import genai
from google.genai import types
import shutil

app = typer.Typer()
env = dotenv_values()


max_output_tokens = 50


def save_torch(datasets: list[FontDatasetEntry], idx: int):
    dataset_path = ROOT_DIR / "data" / cfg.dataset_path / f"{idx}_fonts.ds"
    print(f"Saving datasets to {dataset_path}")
    torch.save(datasets, dataset_path)


def parse_font_to_tensor(t: tuple[int, Series, Path]):
    codec = FontCodec()
    text_embedder = TextEmbedder()
    idx, row, raw_dataset_path = t[0], t[1], t[2]
    try:
        ## Font parsing
        fontfile = raw_dataset_path / row["name"]
        logging.info(f"Parsing font {fontfile}")
        fonts = Fonts(path=fontfile)
        font_paths = (
            fonts.process_check().record(process_check=True).scale_offset().paths()
        )
        font_tokens_dict = codec.encode_font(font_paths)
        splitted_font_tokens = codec.split_font_by_glyphs(
            font_tokens_dict=font_tokens_dict,
            num_combined_glyphs=cfg.num_combined_glyphs,
        )
        # Text parsing
        row = row.drop("name")
        text = ",".join(row.values)

        text_embeddings, text_attn_mask_ids = text_embedder.get_text_emeddings(text)

        return [
            FontDatasetEntry(
                font_tokens.to(device="cpu"),
                text_embeddings[0].to(device="cpu"),
                text_attn_mask_ids[0].bool().to(device="cpu"),
            )
            for font_tokens in splitted_font_tokens
        ]

    except FontGlyphMissing as err:
        logging.debug(err)
        pass
    except MissingBoundary as err:
        logging.debug(err)
        pass
    except MissingDim as err:
        logging.debug(err)
        pass
    except TooManyTokens as err:
        logging.debug(err)
        pass
    except TTLibError as err:
        logging.debug(err)
        pass
    except NotSupportedYet as err:
        logging.debug(err)
        pass
    except BaseException as err:
        logging.error(err)
        pass


@app.command("generate_tensors")
def main(raw_dataset: str, desc_path: str):
    logging.basicConfig(level=logging.INFO)
    cfg.device = "cpu"

    datasets: list[FontDatasetEntry] = []

    chunk_size = 250
    idx = 0

    shutil.rmtree(ROOT_DIR / "data" / cfg.dataset_path, ignore_errors=True)
    (ROOT_DIR / "data" / cfg.dataset_path).mkdir(parents=True, exist_ok=True)

    raw_dataset_path = Path(raw_dataset)
    df_desc = pd.read_csv(Path(desc_path) / "_df_desc.csv")

    with Pool() as p:
        for dataset in tqdm.tqdm(
            p.imap_unordered(
                func=parse_font_to_tensor,
                iterable=[
                    (idx, row, raw_dataset_path)
                    for row_idx, row in df_desc.iloc[0 : len(df_desc), :].iterrows()
                ],
                chunksize=chunk_size,
            ),
            total=len(df_desc),
        ):
            if dataset is not None:
                datasets += dataset
            if len(datasets) >= chunk_size:
                save_torch(datasets=datasets, idx=idx)
                idx += 1
                datasets = []
    idx += 1
    save_torch(datasets=datasets, idx=idx)
    print("All dataset saved!")


def get_prompt(font_name: str):
    return f"For the font with the name {font_name}, please give me one adjective that best represent it. One word only without punctuation."


def get_system_instruction():
    return "You are an expert font designer. You have 20 years of experience in designing fonts. Do not hallucinate."


def gemma_fetch_write(
    file: Path,
    output_dir: Path,
):
    gemma_client = genai.Client(api_key=env["GEMMA_KEY"])
    if file.name.endswith(".ttf"):
        filename_f = file.name
        filename = filename_f.removesuffix(".ttf").replace("_", " ")
        logging.info(f"openai request for {filename}")

        response = gemma_client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=get_prompt(filename),
            config=types.GenerateContentConfig(
                system_instruction=get_system_instruction(),
                max_output_tokens=max_output_tokens,
            ),
        )
        save_filename: str = filename_f.removesuffix(".ttf") + ".txt"
        file_content = response.text
        save_loc = output_dir / save_filename
        if file_content is not None:
            with save_loc.open("w") as fw:
                fw.write(file_content)


@app.command("gemma")
def create_gemma_responses(
    data_dir: str,
    out_dir: str,
):
    """Batch process Gemma requests."""
    output_dir = Path(out_dir)
    dataset_dir = Path(data_dir)

    existing_files = [file.name.removesuffix(".txt") for file in output_dir.iterdir()]

    with Pool() as p:
        p.starmap(
            gemma_fetch_write,
            [
                (file, output_dir)
                for file in dataset_dir.iterdir()
                if file.name.removesuffix(".ttf") not in existing_files
            ],
        )


@app.command("gpt")
def create_gpt_responses(
    data_dir: str,
    out_dir: str,
):
    gpt_client = OpenAI(
        organization=env["GPT_ORG"], project=env["GPT_PRJ"], api_key=env["GPT_KEY"]
    )

    jsonl_file_dir = Path(out_dir)
    dataset_dir = Path(data_dir)
    # for file in jsonl_file_dir.iterdir():
    #     file.unlink()
    for idx, file in enumerate(dataset_dir.iterdir()):
        if file.name.endswith(".ttf"):
            filename_f = file.name
            filename = filename_f.removesuffix(".ttf").replace("_", " ")
            logging.info(f"openai request for {filename}")

            response = gpt_client.responses.create(
                model="gpt-4.1-mini",
                input=[
                    {
                        "role": "developer",
                        "content": "You are an expert font designer. You have 20 years of experience in designing fonts. Do not hallucinate.",
                    },
                    {
                        "role": "user",
                        "content": f"For the font with name {filename}, give me the following attributes comma-delimited, in one line, without headers and all in text. Aperture, Weight, Terminals, Serifs, Stems, Stroke contrast, Axis, Stress, 5 adjectives",
                    },
                ],
                max_output_tokens=50,
            )
            save_filename: str = filename + ".txt"
            file_content = response.output_text
            save_loc = jsonl_file_dir / save_filename
            with save_loc.open("w") as fw:
                fw.write(file_content)


def create_gpt_jsonl(jsonl_dir: str, data_dir: str):
    jsonl_file_dir = Path(jsonl_dir)
    dataset_dir = Path(data_dir)
    for file in jsonl_file_dir.iterdir():
        file.unlink()

    batch_size = 10000
    for idx, file in enumerate(dataset_dir.iterdir()):
        file_idx = idx // batch_size
        jsonl_file = jsonl_file_dir / f"openai_jsonl_batch_{file_idx}.jsonl"

        if file.name.endswith(".ttf"):
            filename_f = file.name
            filename = filename_f.removesuffix(".ttf")

            with jsonl_file.open("a", encoding="utf-8") as f:
                json_dict = {
                    "custom_id": f"request-{idx}-{filename_f}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "o4-mini-2025-04-16",
                        "messages": [
                            {
                                "role": "developer",
                                "content": "You are an expert font designer. You have 20 years of experience in designing fonts. Do not hallucinate.",
                            },
                            {
                                "role": "user",
                                "content": f"For the font {filename}, give me the following attributes comma-delimited as short as possible without headers. Aperture, Weight, Terminals, Serifs, Stems, Stroke contrast, Axis, Stress, 5 adjectives",
                            },
                        ],
                        "max_tokens": 50,
                    },
                }

                f.write(
                    json.dumps(json_dict, indent=4).replace("\n", "").replace("  ", "")
                    + "\n",
                )


@app.command()
def post_gpt_jsonl(jsonl_dir: str, data_dir: str):
    jsonl_file_dir = Path(jsonl_dir)
    dataset_dir = Path(data_dir)

    # Create Batch
    batches = []
    failed_batches = []
    for file in jsonl_file_dir.iterdir():
        logging.info(file)
        batch_input_file = client.files.create(file=open(file, "rb"), purpose="batch")
        batch_input_file_id = batch_input_file.id
        c_batch = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"Create keywords eval job {file}"},
        )
        batches.append(c_batch.id)
        # with open("output.txt", "w") as file:
        #     for item in batches:
        #         file.write(f"{item}\n")
        # while len(batches) > 0:
        print("fetching...")
        while True:
            # for c_batch in batches:
            batch = client.batches.retrieve(c_batch.id)

            if batch.status == "completed" and batch.output_file_id:
                file_response = client.files.content(batch.output_file_id)
                print(file_response.text)

                for jsonl in file_response.text.splitlines():
                    response = json.loads(jsonl)
                    save_filename: str = (
                        response["custom_id"].split("-", 2)[2].replace(".ttf", ".txt")
                    )
                    file_content = response["response"]["body"]["choices"][0][
                        "message"
                    ]["content"]
                    save_loc = dataset_dir / save_filename
                    with save_loc.open("w") as fw:
                        fw.write(file_content)

                batches.remove(batch.id)
                break
            elif batch.status == "failed":
                logging.info("Failed", batch.id)
                failed_batches.append(batch.id)
                logging.info("failed_batches", failed_batches)
                break
            else:
                pass

            time.sleep(600)


def cancel_batch():
    batches = client.batches.list()
    all_batches = {}
    for batch in batches:
        if batch.status == "in_progress":
            logging.info(batch)
            client.batches.cancel(batch.id)


@app.command("collect")
def collect_txt(data_dir: str, out_dir: str):
    data_d = Path(data_dir)
    out_d = Path(out_dir)

    csv_arr = []

    for idx, file in enumerate(data_d.iterdir()):
        if file.suffix == ".txt":
            csv_arr.append(
                {
                    "name": file.name.removesuffix(".txt") + ".ttf",
                    "adjective": file.read_text().strip(),
                }
            )

    pd.DataFrame(csv_arr).to_csv(out_d / "_df_desc.csv", index=False)
    logging.info("CSV saved!")
