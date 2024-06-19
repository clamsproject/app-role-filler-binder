from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from collections import defaultdict

tokenizer = AutoTokenizer.from_pretrained("clamsproject/bert-base-cased-ner-rfb")
model = AutoModelForTokenClassification.from_pretrained("clamsproject/bert-base-cased-ner-rfb")


def parse_sequence_tags(phrases, scene_type):
    """
    Parses an RFB string and returns a dictionary of role-fillers.

    Args:
        phrases (list[str]): An list of tag-token tuples
        scene_type (str): The scene classification label (e.g. "credits")
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

        return dict(bindings)

    # If parsing doesn't work, the string is either invalid or empty (EAFP)
    except Exception as e:
        return {}


def bind_role_fillers(ocr_results, scene_type):
    """
    Runs model from a given checkpoint on OCR results and returns BIO-annoted string.

    Args:
        ocr_results (str): OCR results from a video frame.
        scene_type (str): The type of scene, either "credits" or "chyron".
    """

    rfb_sent = f"{scene_type} {ocr_results}"

    tagger = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    outputs = tagger(rfb_sent)
    words = [(entry["entity_group"], entry["word"]) for entry in outputs if not entry["word"].startswith("##")]
    return parse_sequence_tags(words, scene_type)
