import pytest

from crag.asr.evaluate import wer_by_condition
from crag.asr.metrics import ErrorCount, char_errors, corpus_rate, normalize, word_errors
from crag.asr.transcribe import transcribe_all, transcribe_file


def test_normalize_lowercases_and_strips_punctuation():
    assert normalize("Je, kuna FARASI?") == "je kuna farasi"
    assert normalize("  mwaka   1960.  ") == "mwaka 1960"


def test_word_errors_counts_token_substitutions():
    count = word_errors("a b c d", "a x c d")  # one substitution of four words
    assert count == ErrorCount(edits=1, length=4)
    assert count.rate == 0.25


def test_word_errors_ignores_punctuation_and_case_via_normalisation():
    # Gold has punctuation/case; ASR is lowercase and unpunctuated -> no errors.
    assert word_errors("Je, kuna farasi?", "je kuna farasi").rate == 0.0


def test_char_errors_are_character_level():
    assert char_errors("abc", "abd") == ErrorCount(edits=1, length=3)


def test_empty_reference_has_zero_rate():
    assert word_errors("", "anything").rate == 0.0


def test_corpus_rate_is_micro_averaged():
    # 1/2 and 1/4 micro-average to 2/6, not the macro mean of the two rates.
    counts = [ErrorCount(1, 2), ErrorCount(1, 4)]
    assert corpus_rate(counts) == pytest.approx(2 / 6)


def test_wer_by_condition_aggregates_and_skips_empty_reference():
    items = [
        {"gold_question": "a b c d", "t": {"kenya-sw-KE": "a x c d", "tanzania-sw-KE": "a b c d"}},
        {"gold_question": "e f", "t": {"kenya-sw-KE": "e f", "tanzania-sw-KE": "e z"}},
        {"gold_question": "   ", "t": {"kenya-sw-KE": "ignored"}},  # empty reference -> skipped
    ]
    results = wer_by_condition(items, transcripts_of=lambda item: item["t"])
    assert results["kenya-sw-KE"] == {
        "n": 2, "wer": 0.1667, "cer": results["kenya-sw-KE"]["cer"],
        "word_edits": 1, "ref_words": 6,
        "char_edits": results["kenya-sw-KE"]["char_edits"], "ref_chars": results["kenya-sw-KE"]["ref_chars"],
    }
    assert results["tanzania-sw-KE"]["word_edits"] == 1  # one substitution in the second item


class _Segment:
    def __init__(self, text):
        self.text = text


def test_transcribe_file_joins_segment_texts():
    class FakeModel:
        def transcribe(self, path, language):
            assert language == "sw"
            return [_Segment(" habari"), _Segment(" za asubuhi")], {"language": language}

    assert transcribe_file(FakeModel(), "x.wav") == "habari za asubuhi"


def _fake_transcriber(model, wav, language):
    return f"nakala {wav.name}"


def test_transcribe_all_caches_and_resumes(tmp_path):
    cache = tmp_path / "whisper.json"
    wavs = {1: tmp_path / "1.wav", 2: tmp_path / "2.wav"}
    out = transcribe_all(wavs, cache, model=None, transcriber=_fake_transcriber)
    assert out == {1: "nakala 1.wav", 2: "nakala 2.wav"}
    assert cache.exists()

    def _boom(model, wav, language):
        raise AssertionError("should not re-transcribe a cached id")

    # Same ids again: everything is cached, so the transcriber must not be called.
    again = transcribe_all(wavs, cache, model=None, transcriber=_boom)
    assert again == {1: "nakala 1.wav", 2: "nakala 2.wav"}
