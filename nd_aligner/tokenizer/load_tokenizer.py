from .arpa_tokenizer import ARPATokenizer
from .espeak_tokenizer import ESPEAKTokenizer


def load_tokenizer(tokenizer_type: str, fastspeech2_lexicon_path: str = ""):
    if tokenizer_type == "espeak":
        return ESPEAKTokenizer()
    elif tokenizer_type == "arpa":
        lexicon_path = None if not fastspeech2_lexicon_path else fastspeech2_lexicon_path
        return ARPATokenizer(lexicon_path=lexicon_path)
    else:
        raise ValueError()
