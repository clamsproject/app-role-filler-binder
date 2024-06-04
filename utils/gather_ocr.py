"""
Gather necessary information from SWT and DocTR mmif annotations for RFB.
"""

import json
import pandas as pd
import pathlib
from mmif import Mmif, DocumentTypes, AnnotationTypes
from tqdm import tqdm
import argparse


def gather_ocr_data(data_dir: str) -> list[tuple[str, int, str, float, str]]:
    """
    Takes a directory of mmif files with views from SWT and DocTR.
    Iterates over each TextDocument in the DocTR view, and obtains the corresponding SWT label via Alignments.
    Returns a list of tuples, where each tuple contains the guid, scene, and ocr text.

    :param data_dir: directory containing mmif files
    :return: list of tuples in the form [(guid, scene, ocr text), ...]
    """
    path = pathlib.Path(data_dir)
    outputs = []
    for filename in tqdm(list(path.glob('*.mmif'))):
        with open(filename, 'r') as f:
            curr_mmif = json.load(f)
            curr_mmif = Mmif(curr_mmif)
        guid = filename.stem.split('.')[0]
        swt_view = curr_mmif.get_view_by_id('v_0')
        doctr_view = curr_mmif.get_view_by_id('v_3')  # in my batch chyrons are in 'v_2', credits in 'v_3'
        timeframes = swt_view.get_annotations(at_type=AnnotationTypes.TimeFrame)
        timepoints2frames = {tp_rep: tf for tf in timeframes for tp_rep in tf.get('representatives')}
        timepoints = list(swt_view.get_annotations(at_type=AnnotationTypes.TimePoint))
        timepoints = {tp.get('id'): tp.get('timePoint') for tp in timepoints}
        for textdoc in doctr_view.get_documents():
            ocr_text = rf'{textdoc.text_value}'
            td_id = textdoc.id
            td_alignment = list(doctr_view.get_annotations(AnnotationTypes.Alignment, target=td_id))
            timepoint = td_alignment[0].get('source')
            tp_id = timepoint.split(':')[1]
            timepoint = timepoints[tp_id]
            scene_label = timepoints2frames[tp_id].get('label')
            confidence = timepoints2frames[tp_id].get('classification')[scene_label]
            outputs.append((guid, timepoint, scene_label, confidence, ocr_text))
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mmif_dir', type=str, help='Path to directory containing MMIF files.', required=True)
    parser.add_argument('--drop_duplicates', action='store_true', help='Remove duplicate ocr datapoints')
    parser.add_argument('--scene_types', nargs='+', default=['credits', 'credit', 'chyron'],
                        help='SWT labels to keep. Choices: {bars, slate, chyron, credits}')
    args = parser.parse_args()

    if pathlib.Path(args.mmif_dir).is_dir():
        mmif_dir = pathlib.Path(args.mmif_dir)
        data = gather_ocr_data(args.mmif_dir)
        swt_stitch_doctr_df = pd.DataFrame(data=data,
                                           columns=['guid', 'timePoint', 'scene_label', 'confidence', 'textdocument'])
        if args.drop_duplicates:
            swt_stitch_doctr_df = swt_stitch_doctr_df.drop_duplicates(subset=['textdocument'])
        # filter SWT labels
        swt_stitch_doctr_df = swt_stitch_doctr_df[swt_stitch_doctr_df['scene_label'].isin(args.scene_types)]
        swt_stitch_doctr_df.to_csv('doctr-preds-credits.csv', index=False)
    else:
        print('Please provide a valid directory path.')
