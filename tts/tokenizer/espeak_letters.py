PUNCTUATION = ';:,.!?¡¿—…"«»“” '
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
LETTERS_IPA = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

BOS = "^"
EOS = "~"

SYMBOLS = [BOS, EOS] + list(PUNCTUATION) + list(LETTERS) + list(LETTERS_IPA)

SYMBOL_DICTS = {}
for i in range(len((SYMBOLS))):
    SYMBOL_DICTS[SYMBOLS[i]] = i


IGNORE_SYMBOLS = {
    "^",
    "~",
    ";",
    ":",
    ",",
    ".",
    "!",
    "?",
    "¡",
    "¿",
    "—",
    "…",
    '"',
    "«",
    "»",
    "“",
    "”",
}
