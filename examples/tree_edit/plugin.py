"""External parser + zss ordered tree edit distance. Trees are (label, children)."""
from uncertain import RepresentationSet


class TreeEncoder:
    output_type = "tree"
    supported_units = {"token", "word", "sentence"}

    def __init__(self, parse, *, parser_id, backend):
        self.parse, self.backend = parse, backend
        self.descriptor = {"encoder": "example:tree", "version": "1", "parser_id": parser_id,
                           "backend": backend.descriptor}

    def encode(self, source, short_samples):
        if short_samples == "pad":
            raise ValueError("tree example requires drop or compare_ragged")
        return RepresentationSet(tuple(self.parse(self.backend.decode(s.tokens)) for s in source.sequences),
                                 source, self.descriptor, self.output_type)


class TreeEdit:
    input_type = "tree"
    range = (0., float("inf"))
    kind = "dissimilarity"
    negative_type = False
    symmetric = True

    def __init__(self):
        from importlib.metadata import version
        self.descriptor = {"distance": "example:tree_edit", "version": "1", "zss": version("zss")}

    def __call__(self, a, b):
        from zss import simple_distance
        return float(simple_distance(a, b, get_children=lambda node: node[1],
                                     get_label=lambda node: node[0],
                                     label_dist=lambda x,y: int(x != y)))
