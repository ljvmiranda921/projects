import spacy
from spacy.tokens import Doc
from spacy.vocab import Vocab
from spacy.training import biluo_tags_to_offsets
from spacy.training import iob_to_biluo
import srsly
from pathlib import Path
from prodigy.util import set_hashes
from prodigy.components.preprocess import add_tokens
from scripts.rules import restaurant_span_rules
import copy

import typer

Arg = typer.Argument
Opt = typer.Option


def get_text_annotations(
    file_in: Path = Arg(..., help="Input path for the raw IOB files."),
):
    """
    The IOB format from the MIT Restaurant reviews dataset has the tokens and
    its annotations on individual lines. Here we turn the tokens and annotations
    into a json-format dataset and also save just the text for other functions.

    Returns a dictionary with text, spans, and annotator ID and a dictionary with
    text.
    """
    # open IOB data
    with file_in.open("r", encoding="utf-8") as infile:
        input_lines = infile.read().splitlines()

    # the annotations and tokens are separated by '\t'
    annotation_token = [line.split("\t") for line in input_lines]

    text = []  # the string for each entry
    tags = []  # the labels for each entry
    data = []  # the full list of jsonl data with annotations
    text_data = []  # the list of jsonl data without annotations

    for lines in annotation_token:
        if len(lines) == 2:  # contains both annotation and token
            annotation, token = lines
            text.append(token)
            tags.append(annotation)
        if len(lines) == 1:  # contains space
            doc = Doc(Vocab(), words=text)
            tags = iob_to_biluo(tags)
            offsets = biluo_tags_to_offsets(doc, tags)  # (0, 7, Rating)
            # create ents from offsets
            ents = [
                {"start": span[0], "end": span[1], "label": span[2]} for span in offsets
            ]
            # append data to lists
            data.append(
                {
                    "text": " ".join(text),
                    "spans": ents,
                    "_annotator_id": "original_annotations",
                    "_session_id": "original_annotations",
                }
            )
            text_data.append({"text": " ".join(text)})
            # clear lists
            text = []
            tags = []

    return data, text_data


def get_model_data(
    text_data=Arg(..., help="Dictionary of texts in the dataset."),
    trained_model=Arg(..., help="The trained NER model."),
):
    """
    Create JSON data with model annotations from the trained NER model.

    Returns a dictionary with text, spans, and annotator ID.
    """
    # load trained model
    nlp = spacy.load(trained_model)

    data = copy.deepcopy(text_data)
    for line in data:
        text = line["text"]

        doc = ner(text)
        spans = []
        for ent in doc.ents:
            # Create a span dict for the predicted entity.
            spans.append(
                {"start": ent.start_char, "end": ent.end_char, "label": ent.label_,}
            )
        line["spans"] = spans
        line["_annotator_id"] = "ner_model"
        line["_session_id"] = "ner_model"

    return data


def get_ruler_data(text_data=Arg(..., help="Dictionary of texts in the dataset."),):
    """
    Create JSON data with annotations from the SpanRuler patterns.

    Returns a dictionary with text, spans, and annotator ID.
    """
    nlp = spacy.blank("en") 

    # add span ruler pattern pipe on blank tokenizer
    patterns = restaurant_span_rules()
    ruler = nlp.add_pipe("span_ruler")
    ruler.add_patterns(patterns)

    data = copy.deepcopy(text_data)
    for line in data:
        text = line["text"]
        doc = nlp(text)
        spans = []
        for span in doc.spans["ruler"]:
            # Create a span dict for the predicted entity.
            spans.append(
                {"start": span.start_char, "end": span.end_char, "label": span.label_,}
            )
        line["spans"] = spans
        line["_annotator_id"] = "ruler"
        line["_session_id"] = "ruler"

    return data


def preprocess_prodigy(
    input_file: Path = Arg(..., help="Input path for the raw IOB files."),
    output_file: Path = Arg(..., help="Output path for the processed jsonl files."),
    trained_model: Path = Arg(..., help="The trained NER model."),
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
    data_annotations, data_text = get_text_annotations(
        file_in
    )  # original annotations, text data
    data_model = get_model_data(data_text, trained_model)  # NER model annotations

    # combine data
    data = data_annotations + data_model

    if include_ruler:
        data_ruler = get_ruler_data(data_text)  # SpanRuler pattern annotations
        data += data_ruler

    nlp = spacy.blank("en")  # blank tokenizer
    stream = add_tokens(nlp=nlp, stream=data, skip=True)  # add tokens to data
    # set hashes in stream only on text key so they're the same across examples
    stream = (
        set_hashes(
            eg, task_keys=("text"), ignore=("spans", "_annotator_id", "_session_id")
        )
        for eg in stream
    )

    # write to file
    srsly.write_jsonl(file_out, stream)


if __name__ == "__main__":
    typer.run(preprocess_prodigy)
