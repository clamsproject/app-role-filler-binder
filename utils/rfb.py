from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from collections import defaultdict
from typing import List

tokenizer = AutoTokenizer.from_pretrained("clamsproject/bert-base-cased-ner-rfb")
model = AutoModelForTokenClassification.from_pretrained("clamsproject/bert-base-cased-ner-rfb")
tagger = pipeline("token-classification", model=model, tokenizer=tokenizer, device_map="auto",
                  aggregation_strategy="first")


def parse_sequence_tags(phrases, scene_type) -> List[dict]:
    """
    Parses a list of NER label-mention pairs and returns a dictionary of role-fillers.

    Args:
        phrases (List[tuple[str, str]): A list of tag-mention tuples
        scene_type (str): The scene classification label (e.g. "credits")

    Returns:
        List[dict]: A list of dictionaries containing role-fillers in format [{"Role": role, "Filler": filler}]
    """

    try:
        start_phrase = "ROLE" if scene_type.lower() == "credits" else "FILL"
        bindings = defaultdict(list)
        cur_role = ""
        cur_fillers = []

        for phrase in phrases:
            role, word = phrase
            if role == start_phrase:
                if cur_role or cur_fillers:
                    bindings[cur_role] = cur_fillers if cur_role not in bindings \
                        else bindings[cur_role] + cur_fillers
                cur_role = ""
                cur_fillers = []

            if role == "ROLE":
                cur_role = word
            elif role == "FILL":
                cur_fillers.append(word)

        if cur_role or cur_fillers:
            bindings[cur_role] = cur_fillers if cur_role not in bindings \
                else bindings[cur_role] + cur_fillers

        binding_pairs = [{"Role": role, "Filler": filler} for role, fillers in bindings.items() for filler in fillers]
        # Account for empty fillers
        binding_pairs += [{"Role": role, "Filler": ""} for role, fillers in bindings.items() if len(fillers) == 0]
        return binding_pairs

    # If parsing doesn't work, the string is either invalid or empty (EAFP)
    except Exception as e:
        return {}


def bind_role_fillers(ocr_results, scene_type, clf=tagger):
    """
    Runs model from a given checkpoint on OCR results and returns BIO-annotated string.

    Args:
        clf (Pipeline): A HuggingFace pipeline for Token Classification
        ocr_results (str): OCR results from a video frame.
        scene_type (str): The type of scene, either "credits" or "chyron".
    """

    rfb_sent = f"{scene_type} {ocr_results}"

    outputs = clf(rfb_sent)
    words = [(entry["entity_group"], entry["word"]) for entry in outputs]
    return parse_sequence_tags(words, scene_type)
