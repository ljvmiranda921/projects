<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Using SpanRuler for rule-based Named Entity Recognition

This example project demonstrates how you can use the
[SpanRuler](https://spacy.io/api/spanruler) component for rule-based named
entity recognition (NER). In spaCy v3 and below, this functionality can be
achieved via the [EntityRuler](https://spacy.io/api/entityruler). However, 
we will start deprecating the `entity_ruler` component in v4 in favor of
`span_ruler`.

Here, we will be using the **MIT Restaurant dataset** (Liu, et al, 2013) to
determine entities such as *Rating*, *Location*, *Restaurant_Name*,
*Price*, *Hours*,  *Dish*, *Amenity*,  and *Cuisine* from restaurant reviews.
Below are a few examples from the training data:


First, we will train an NER model and treat it as our baseline. Then, we will
attach the `SpanRuler` component before and after the existing pipeline. This
setup gives us three pipelines we can compare upon.

Here are some set of rules we included in the patterns file (`patterns.jsonl`):

| Label  | Pattern / Examples                                    | Description                                                                 |
|--------|-------------------------------------------------------|-----------------------------------------------------------------------------|
| Price  | `cheap(est)?`, `(in)?expensive`, `decent(ly)?`, `affordable`  | Reviewers do not often give the exact dollar amount but rather describe it. |
| Price  | `\d+\sdollar(s)?                                        | Reviewers, in some cases, give the exact price in dollars.                  |
| Rating | `good`, `great`, `fancy`                                           | Reviewers often describe the dish rather than giving an exact rating        |
| Rating | `\d(-|\s)?star(s)?`                                     | Reviewers can also give star ratings (5-star, 3-stars, 2 star) on a dish.   |
| Rating | `(one|two|three|four|five)\sstar(s)?`                  | Same as above but using words instead of numbers. |

- J. Liu, P. Pasupat, S. Cyphers, and J. Glass. 2013. Asgard: A portable
architecture for multilingual dialogue systems. In *2013 IEEE International
Conference on Acoustics, Speech and Signal Processing*, pages 8386-8390


## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[spaCy projects documentation](https://spacy.io/usage/projects).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `download` | Download a spaCy model with pretrained vectors. |
| `preprocess` | Format and process the raw IOB datasets to make them compatible with spaCy convert. |
| `convert` | Convert the data to spaCy's binary format. |
| `split` | Split the train-dev dataset. |
| `train` | Train a baseline NER model. |
| `attach-rules` | Attach rules to the baseline NER model. |
| `evaluate` | Evaluate each model. |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `download` &rarr; `preprocess` &rarr; `convert` &rarr; `split` &rarr; `train` &rarr; `attach-rules` &rarr; `evaluate` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/train_raw.iob` | URL | Training data from the MIT Restaurants Review dataset |
| `assets/test_raw.iob` | URL | Test data from the MIT Restaurants Review dataset |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->