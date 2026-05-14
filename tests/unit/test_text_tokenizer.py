import pytest
import torch

from tts.tokenizer.text_tokenizer import TextTokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return TextTokenizer()


def test_text_tokenizer_workflow(
    tokenizer: TextTokenizer,
    visualize: bool,
):
    s = "hello world!"

    ipa_text = tokenizer.to_ipa(s)
    assert isinstance(ipa_text, str)
    assert len(ipa_text) > 0

    tokens = tokenizer(s)
    assert isinstance(tokens, torch.Tensor)
    assert tokens.shape[0] == 1  # Batch size 1 확인
    assert tokens[0, 0].item() == 0  # PAD 토큰 확인

    print(f"Input Text:  {s}")
    print(f"IPA Phonemes: {ipa_text}")
    print(f"Token IDs:    {tokens.tolist()}")
    print(f"Vocab Size:   {tokenizer.n_vocab}")


def test_consistency(tokenizer: TextTokenizer):
    s = "Consistency test."
    assert tokenizer.to_ipa(s) == tokenizer.to_ipa(s)
    assert torch.equal(tokenizer(s), tokenizer(s))


@pytest.mark.parametrize(
    "text",
    [
        "How's the weather today?",
        "Check-out these numbers: 1, 2, 3.",
        "Special characters like @#$% might be ignored or handled.",
    ],
)
def test_various_sentences(
    tokenizer: TextTokenizer,
    text: str,
    visualize: bool,
):
    tokens = tokenizer(text)
    assert tokens.dim() == 2

    print(f"\nInput: {text}")
    print(f"IPA:   {tokenizer.to_ipa(text)}")
