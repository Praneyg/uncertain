"""Inject a pinned tagger; spaCy is an optional adapter, never a core dependency."""
from uncertain import RepresentationSet


class POSEncoder:
    output_type = "units"
    supported_units = {"token", "word", "sentence"}

    def __init__(self, tagger, *, tagger_id, backend):
        self.tagger, self.backend = tagger, backend
        self.descriptor = {"encoder": "example:pos", "version": "1", "tagger_id": tagger_id,
                           "backend": backend.descriptor}

    def encode(self, source, short_samples):
        if short_samples == "pad":
            raise ValueError("POS example requires drop or compare_ragged")
        reps = tuple(tuple(self.tagger(self.backend.decode(row.tokens))) for row in source.sequences)
        return RepresentationSet(reps, source, self.descriptor, self.output_type)


def spacy_encoder(model_name, *, model_revision, backend):
    import spacy
    pipeline = spacy.load(model_name)
    return POSEncoder(lambda text: [token.pos_ for token in pipeline(text)],
                      tagger_id=f"{model_name}@{model_revision}", backend=backend)
