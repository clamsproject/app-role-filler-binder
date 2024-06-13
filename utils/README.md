### `prepare_data.py`
Converts the data set annotated by the Claude 3 Haiku LLM into a format ready for training the BERT model to perform token classification. 
The "raw" annotated data is available at [`aapb-annotations` repo](https://github.com/clamsproject/aapb-annotations/tree/c1fb0287b4e70f9ce8d271759e5af4d63eb9a32f/role-filler-binding-seqtag/240605-aapb-annotation-44).

To prepare the model inputs, the script prepends the corresponding scene label to each ocr text sequence before splitting the sequence into a list of tokens.  To prepare the labels, for each ocr text sequence annotated by Haiku (format: `token@tag:index`), all the tags are extracted and compiled into a list the same length as the token sequence. The end result is a dataframe consisting of just the text tokens and their corresponding tags. Finally, this dataframe is shuffled and partitioned into train, validation, and test splits (8:1:1 ratio) in JSON format.

The prepared data is saved in the `model_in_data` directory as train/val/dev split JSON files.
