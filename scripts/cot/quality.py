"""Deterministic review flags and text lengths; these do not verify visual truth."""
import re

_EVIDENCE = re.compile(r'<evidence>.*?</evidence>', re.S)
_SOURCE_REFERENCE = re.compile(
    r"\b(?:the\s+)?annotator(?:s)?\s+(?:says?|notes?|reports?|describes?|identified)\b"
    r"|\b(?:according to|based on|as stated in|as described in)\s+(?:the\s+)?(?:human\s+)?(?:transcript(?:ion)?|annotations?|reference answer|ground truth)\b"
    r"|\b(?:the\s+)?(?:transcript(?:ion)?|reference answer|ground truth)\s+(?:says?|states?|indicates?|shows?|confirms?|specifies?)\b"
    r"|\b(?:human[- ]annotated|annotator[- ]provided)\s+(?:points?|labels?|evidence|answers?)\b"
    r"|\b(?:supplied|provided|reference|ground[- ]truth)\s+(?:labels?|answers?)\b",
    re.I,
)


def review_response(response):
    flags, statistics = [], {}
    for field in ('answer', 'reasoning_process'):
        text = response.get(field)
        if not isinstance(text, str):
            continue
        prose = _EVIDENCE.sub(' ', text)
        statistics[field] = {'characters': len(text), 'prose_characters': len(prose.strip()),
                             'prose_words': len(prose.split())}
        for match in _SOURCE_REFERENCE.finditer(prose):
            flags.append({'type': 'annotation_source_reference', 'field': field, 'match': match.group(0)})
    return {'source_reference_check': 'flagged' if flags else 'no_flags',
            'flags': flags, 'text_statistics': statistics}
