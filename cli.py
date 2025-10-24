import typer
import fontweaver.sampler as sampler

import fontweaver.process_dataset as process
import fontweaver.train as train
import logging

app = typer.Typer()
app.add_typer(sampler.app, name="sample")
app.add_typer(process.app, name="process")
app.add_typer(train.app, name="train")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app()
