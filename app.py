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
from mmif.utils import sequence_helper as sqh

# For an NLP tool we need to import the LAPPS vocabulary items
from lapps.discriminators import Uri

import metadata

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

        # TODO: Add support for user-defined labels.
        #  However, they MUST map to tokens the RFB model has been trained on.
        labelmap = {'I': 'chyron', 'N': 'chyron', 'Y': 'chyron', 'C': 'credits', 'R': 'credits'}

        rfb_view = mmif.new_view()
        self.sign_view(rfb_view, parameters)
        rfb_view.new_contain(DocumentTypes.TextDocument)
        rfb_view.new_contain(AnnotationTypes.Alignment)

        for view in mmif.get_all_views_contain(AnnotationTypes.TimePoint):
            for tp_ann in view.get_annotations(AnnotationTypes.TimePoint):
                for aligned in tp_ann.get_all_aligned():
                    if aligned.at_type == DocumentTypes.TextDocument:
                        td_ann = aligned
                        tp_label = tp_ann.get('label')
                        self.logger.debug(f"Found a TextDocument `{td_ann.long_id}`"
                                          f" anchored to TimePoint `{tp_ann.long_id}` labeled `{tp_label}`")
                        if tp_label in labelmap.keys():
                            scene: str = labelmap[tp_label]
                            self.logger.debug(f"Processing {scene.upper()} TextDocument `{td_ann.long_id}` ")
                            ocr_text = rf'{td_ann.text_value}'
                            input_seq = " ".join(clean_ocr(ocr_text))
                            parsed = bind_role_fillers(input_seq, scene)
                            self.logger.debug(f"Found {len(parsed)} Role-Filler pairs.")
                            if not parsed:
                                continue
                            else:
                                csv_string = pd.DataFrame.from_dict(parsed).to_csv()
                                new_doc = rfb_view.new_textdocument(text=csv_string)
                                rfb_view.new_annotation(
                                    at_type=AnnotationTypes.Alignment, source=td_ann.long_id, target=new_doc.long_id
                                )
                                self.logger.debug(
                                    f"Created annotation `{new_doc.long_id}` anchored to `{td_ann.long_id}`"
                                )
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
