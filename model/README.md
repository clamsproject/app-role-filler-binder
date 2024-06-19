# Role-Filler Binder (RFB)

## Installation

Run `pip install requirements.txt` to install app requirements.

## Model

The model used for this task is a bert-base-cased model fine-tuned on a custom dataset. It currently only performs the NER step of role-filler binding, and relies on a follow-up heuristic algorithm for grouping fillers with their respective roles. The current model checkpoint may be loaded from HuggingFace at `clamsproject/bert-base-cased-ner-rfb`.

The model may be traiend by running:

```bash
python run_ner.py args.json
```

Make any changes to `args.json` as needed. Ensure that it is in the same directory when running, and that it correctly points to the relevant train/val/test JSONL files. The script handles both training and evaluation (reporting PRF metrics).
