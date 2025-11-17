## Fine tunning Gemma 3 270M for text to SQL

```bash
$ uv venv
$ source .venv/bin/activate
$ uv sync
```


### For fine tunning
```bash
$ uv run main.py train --hf-token <huggingface token> --checkpoint-dir <path to save checkpoints of model> --visualization-log <path to save image>
```


### For evaluate model
```bash
$ uv run main.py eval --hf-token <huggingface token> --checkpoint-dir <path to save checkpoints of model or model ID>
```

