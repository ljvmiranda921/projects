import copy
from pathlib import Path
from typing import Dict, Union

import spacy
import srsly
import typer
from prodigy.components.preprocess import add_tokens
from prodigy.util import set_hashes
from spacy.tokens import Doc
from spacy.training import biluo_tags_to_offsets, iob_to_biluo
from spacy.vocab import Vocab

from scripts.rules import restaurant_span_rules

Arg = typer.Argument
Opt = typer.Option


def get_text_annotations(
    input_file: Path = Arg(..., help="Input path for the raw IOB files."),
):
    """
    The IOB format from the MIT Restaurant reviews dataset has the tokens and
    its annotations on individual lines. Here we turn the tokens and annotations
    into a json-format dataset and also save just the text for other functions.

    Returns a dictionary with text, spans, and annotator ID and a dictionary with
    text.
    """
    # open IOB data
    with input_file.open("r", encoding="utf-8") as infile:
        input_lines = infile.read().splitlines()

    # the annotations and tokens are separated by '\t'
    annotation_token = [line.split("\t") for line in input_lines]

    text = []  # the string for each entry
    annotations = []  # the labels for each entry
    org_annotations = []  # the full list of jsonl data with annotations
    texts = []  # the list of jsonl data without annotations

    for lines in annotation_token:
        if len(lines) == 2:  # contains both annotation and token
            annotation, token = lines
            text.append(token)
            annotations.append(annotation)
        if len(lines) == 1:  # contains space
            doc = Doc(Vocab(), words=text)
            annotations = iob_to_biluo(annotations)
            offsets = biluo_tags_to_offsets(doc, annotations)  # (0, 7, Rating)
            # create ents from offsets
            ents = [
                {"start": span[0], "end": span[1], "label": span[2]} for span in offsets
            ]
            # append data to lists
            org_annotations.append(
                {
                    "text": " ".join(text),
                    "spans": ents,
                    "_annotator_id": "original_annotations",
                    "_session_id": "original_annotations",
                }
            )
            texts.append({"text": " ".join(text)})
            # clear lists
            text = []
            annotations = []

    return org_annotations, texts


def get_model_data(
    texts: Dict[str, any] = Arg(..., help="Dictionary of texts in the dataset."),
    model: Union[str, Path] = Arg(..., help="The trained NER model."),
):
    """
    Create JSON data with model annotations from the trained NER model.

    Returns a dictionary with text, spans, and annotator ID.
    """
    # load trained model
    nlp = spacy.load(model)

    texts_copy = copy.deepcopy(texts)
    for line in texts_copy:
        text = line["text"]

        doc = nlp(text)
        spans = []
        for ent in doc.ents:
            # Create a span dict for the predicted entity.
            spans.append(
                {
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "label": ent.label_,
                }
            )
        line["spans"] = spans
        line["_annotator_id"] = "ner_model"
        line["_session_id"] = "ner_model"

    return texts_copy


def get_ruler_data(
    texts: Dict[str, any] = Arg(..., help="Dictionary of texts in the dataset."),
):
    """
    Create JSON data with annotations from the SpanRuler patterns.

    Returns a dictionary with text, spans, and annotator ID.
    """
    nlp = spacy.blank("en")

    # add span ruler pattern pipe on blank tokenizer
    patterns = restaurant_span_rules()
    ruler = nlp.add_pipe("span_ruler")
    ruler.add_patterns(patterns)

    texts_copy = copy.deepcopy(texts)
    for line in texts_copy:
        text = line["text"]
        doc = nlp(text)
        spans = []
        for span in doc.spans["ruler"]:
            # Create a span dict for the predicted entity.
            spans.append(
                {
                    "start": span.start_char,
                    "end": span.end_char,
                    "label": span.label_,
                }
            )
        line["spans"] = spans
        line["_annotator_id"] = "ruler"
        line["_session_id"] = "ruler"

    return texts_copy


def preprocess_prodigy(
    input_file: Path = Arg(..., help="Input path for the raw IOB files."),
    output_file: Path = Arg(..., help="Output path for the processed jsonl files."),
    model: Path = Arg(..., help="The trained NER model."),
    include_ruler: bool = Opt(
        False, help="Whether to include the ruler in the outputted annotations."
    ),
):
    """
    Preprocess the raw IOB files from MIT Restaurant Reviews into JSONL with
    the different annotations as multiple annotators.

    Outputs a JSONL file with annotations from the original dataset, the trained
    NER model, and the SpanRuler patterns (optional).
    """
    org_annotations, texts = get_text_annotations(
        input_file
    )  # original annotations, text data
    model_annotations = get_model_data(texts, model)  # NER model annotations

    # combine data
    combined_annotations = org_annotations + model_annotations

    if include_ruler:
        ruler_annotations = get_ruler_data(texts)  # SpanRuler pattern annotations
        combined_annotations += ruler_annotations

    nlp = spacy.blank("en")  # blank tokenizer
    stream = add_tokens(
        nlp=nlp, stream=combined_annotations, skip=True
    )  # add tokens to data
    # set hashes in stream only on text key so they're the same across examples
    stream = (
        set_hashes(
            eg, task_keys=("text"), ignore=("spans", "_annotator_id", "_session_id")
        )
        for eg in stream
    )

    # write to file
    srsly.write_jsonl(output_file, stream)


if __name__ == "__main__":
    typer.run(preprocess_prodigy)
