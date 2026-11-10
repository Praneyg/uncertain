"""Explicit, deliberately simple boundary detectors; no language-model imports."""
import re


class AlignmentError(ValueError):
    pass


def word_boundaries(text, *, final=False):
    """Whitespace-delimited words; the last is unconfirmed until text ends (`final`)."""
    return tuple((m.start(), m.end()) for m in re.finditer(r"\S+", text)
                 if final or m.end() < len(text))


def sentence_boundaries(text, *, final=False):
    """Sentences ending in [.!?]+ followed by whitespace; not abbreviation aware."""
    result = []
    start = 0
    for m in re.finditer(r"[.!?]+(?=\s|$)", text):
        # A terminator at end-of-text is unconfirmed: the next token may extend
        # the sentence ("z?" -> "z?x."). Mirrors the word detector, and keeps
        # units stable under generation so truncate(sample(H_max),H)==sample(H).
        if not final and m.end() >= len(text):
            break
        segment = text[start:m.end()]
        a = start + len(segment) - len(segment.lstrip())
        if a < m.end():
            result.append((a, m.end()))
        start = m.end()
    if final and text[start:].strip():
        a = start + len(text[start:]) - len(text[start:].lstrip())
        result.append((a, len(text.rstrip())))
    return tuple(result)


def boundaries(text, unit, *, final=False):
    """Dispatch to the registered unit's detector, so a new unit needs no core edit."""
    from uncertain.registry import resolve
    detect = getattr(resolve("unit", unit.name), "detect", None)
    if detect is None:
        raise AlignmentError(f"unit {unit.name!r} declares no text boundary detector")
    return detect(text, final=final)


def align(tokens, backend, unit, *, final=False):
    """Return completed units plus token/character spans, allowing token overlap."""
    if unit.name == "token":
        # Token text is decoded by text encoders from exact IDs. This avoids
        # stale text after pure truncation, including byte-token prefix revisions.
        return tuple(str(t) for t in tokens), tuple((i, i+1) for i in range(len(tokens))), (), ""
    text = backend.decode(tokens)
    offsets = [0]
    previous = ""
    for i in range(1, len(tokens)+1):
        current = backend.decode(tokens[:i])
        if i < len(tokens):
            # Byte-level tokenizers (GPT-2) split multi-byte characters; the partial
            # sequence decodes to U+FFFD until a later token completes it. Only the
            # complete characters are stable, so don't count the replacement yet.
            current = current.rstrip("�")
        if not current.startswith(previous):
            raise AlignmentError("decoder revises prior characters; this tokenizer needs a custom aligner")
        offsets.append(len(current))
        previous = current
    chars = boundaries(text, unit, final=final)
    spans = []
    for a, b in chars:
        # A token holding only partial bytes has zero width; it owns the character it starts.
        owners = [i for i in range(len(tokens))
                  if offsets[i] < b and max(offsets[i+1], offsets[i]+1) > a]
        if not owners:
            raise AlignmentError("completed unit has no owning token")
        spans.append((owners[0], owners[-1]+1))
    return tuple(text[a:b] for a,b in chars), tuple(spans), chars, text
