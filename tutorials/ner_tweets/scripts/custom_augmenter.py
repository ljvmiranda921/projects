import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import spacy
from wasabi import msg


@spacy.registry.augmenters("tokensub_augmenter.v1")
def create_augmenter(
    simulate_mistype: bool = True,
    contextual_substitute: bool = True,
):
    def augment(nlp, example):
        msg.text(
            f"Custom augmenter, mistype={simulate_mistype}, subs={contextual_substitute}"
        )
        text = example.text
        msg.text(text)

        # Augment based on a few common mistypes. Don't perturb too much
        if simulate_mistype:
            text = nac.KeyboardAug(aug_char_max=3).augment(text)
            text = nac.RandomCharAug(action="swap", aug_char_max=2).augment(text)
        # Substitute texts based on contextual information
        # uses a BERT model
        if contextual_substitute:
            text = naw.ContextualWordEmbsAug(
                model_path="distilbert-base-uncased", action="substitute"
            ).augment(text)
        msg.text(text)

        example_dict = example.to_dict()
        doc = nlp.make_doc(text)
        example_dict["token_annotation"]["ORTH"] = [t.text for t in doc]
        # FIXME
        msg.text(example_dict)

        yield example
        yield example.from_dict(doc, example_dict)

    return augment
