# Role Filler Binder

## Description

This app implements Role-Filler Binding (RFB) on text using a combination of NER and rule-based string parsing.

Roles and Fillers are analogous to key-value pairs, where the key (role) may correspond to a job title (e.g. Executive Producer),
and the value (filler) corresponds to the named entity mention(s) filling that position. Binding applies an explicit relation between the two.

The current release of RFB uses a [fine-tuned Bert-Base-Cased model](https://huggingface.co/clamsproject/bert-base-cased-ner-rfb)
to classify role/filler tokens and employs heuristic methods to determine relations between them.
The intended target is OCR text extracted from visual media sources using upstream CLAMS apps, and the output is formatted
as a raw CSV string.

## User instruction

General user instructions for CLAMS apps is available at [CLAMS Apps documentation](https://apps.clams.ai/clamsapp).

Below is a list of additional information specific to this app.

### System requirements

- Requires Python3 with `clams-python`, `mmif-python` and their dependencies installed, to run the app locally.
- Requires `transformers` for model inference.
- Requires an HTTP client utility (such as `curl`) to invoke and execute analysis.
- Recommends GPU to run at a reasonable speed.

Run `pip install requirements.txt` to install app requirements.

#### To run in a container

- Requires `docker` to run the app in a Docker container (as an HTTP server).

### Configurable runtime parameter

For the full list of parameters, please refer to the app metadata from [CLAMS App Directory](https://apps.clams.ai) or [`metadata.py`](metadata.py) file in this repository.
