"""
This app implements Role-Filler Binding on text using a combination of NER and rule-based string parsing.
The intended target is OCR text extracted from visual sources using upstream CLAMS apps.
"""

import argparse
import logging

# Imports needed for Clams and MMIF.
# Non-NLP Clams applications will require AnnotationTypes

from clams import ClamsApp, Restifier
from mmif import Mmif, View, Annotation, Document, AnnotationTypes, DocumentTypes

# For an NLP tool we need to import the LAPPS vocabulary items
from lapps.discriminators import Uri

import pandas as pd
from utils.clean_ocr import clean_ocr
from utils.rfb import bind_role_fillers


class RoleFillerBinder(ClamsApp):

    def __init__(self):
        super().__init__()

    def _appmetadata(self):
        # see https://sdk.clams.ai/autodoc/clams.app.html#clams.app.ClamsApp._load_appmetadata
        # Also check out ``metadata.py`` in this directory. 
        pass

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        # see https://sdk.clams.ai/autodoc/clams.app.html#clams.app.ClamsApp._annotate
        self.logger.debug(f"Parameters: {parameters}")
        if not isinstance(mmif, Mmif):
            mmif = Mmif(mmif)
        views2alignments = mmif.get_alignments(AnnotationTypes.TimePoint, DocumentTypes.TextDocument)
        rfb_view = mmif.new_view()
        self.sign_view(rfb_view, parameters)
        rfb_view.new_contain(DocumentTypes.TextDocument)
        rfb_view.new_contain(AnnotationTypes.Alignment)
        for view_id in views2alignments:
            for alignment in views2alignments[view_id]:
                source = alignment.get("source")
                target = alignment.get("target")
                if alignment.id_delimiter not in source:
                    source = alignment.id_delimiter.join((view_id, source))
                if alignment.id_delimiter not in target:
                    target = alignment.id_delimiter.join((view_id, target))
                if {mmif[source].at_type, mmif[target].at_type} != {AnnotationTypes.TimePoint,
                                                                    DocumentTypes.TextDocument}:
                    self.logger.debug(f"Skipping unexpected alignment, source: {source}, target: {target}")
                    continue
                if (mmif[source].at_type == AnnotationTypes.TimePoint
                        and mmif[target].at_type == DocumentTypes.TextDocument):
                    tp, td = mmif[source], mmif[target]
                else:
                    tp, td = mmif[target], mmif[source]
                if tp.get("labelset") is None or not {'I', 'Y', 'N', 'C', 'R'}.issubset(set(tp.get("labelset"))):
                    self.logger.debug(f"Skipping TimePoint with unexpected labelset: {tp.get('labelset')}.")
                    continue
                scene_type = tp.get("label")
                if scene_type in {"I", "Y", "N"}:
                    ftype = "chyron"
                elif scene_type in {"C", "R"}:
                    ftype = "credits"
                else:
                    self.logger.debug(f"Skipping TimePoint `{tp.long_id}` which has an unsupported scene type:"
                                      f" `{scene_type}`")
                    continue
                self.logger.debug(f"Processing {ftype.upper()} TextDocument `{td.long_id}`, anchored to TimePoint"
                                  f" `{tp.long_id}`")
                ocr_text = rf'{td.text_value}'
                input_seq = " ".join(clean_ocr(ocr_text))
                parsed = bind_role_fillers(input_seq, ftype)
                csv_string = pd.DataFrame.from_dict(parsed).to_csv()
                doc = rfb_view.new_textdocument(text=csv_string)
                rfb_view.new_annotation(at_type=AnnotationTypes.Alignment, source=alignment.long_id, target=doc.id)
        return mmif


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")

    parsed_args = parser.parse_args()

    # create the app instance
    app = RoleFillerBinder()

    http_app = Restifier(app, port=int(parsed_args.port))
    # for running the application in production mode
    if parsed_args.production:
        http_app.serve_production()
    # development mode
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
