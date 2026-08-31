from .base_tokenizer import BaseTokenizer
from .espeak_tokenizer import ESPEAKTokenizer


def load_tokenizer(tokenizer_type: str) -> BaseTokenizer:
    if tokenizer_type == "espeak":
        return ESPEAKTokenizer()
    elif tokenizer_type == "espeak_closure":
        return ESPEAKTokenizer(expand_closure_tokens=True)
    else:
        raise ValueError()
