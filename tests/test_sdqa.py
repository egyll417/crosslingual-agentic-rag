import json

from crag.data.io import write_snapshot
from crag.data.sdqa import (
    Gold,
    SDQAItem,
    build_eval_set,
    decode_span,
    parse_gold,
)


def byte_span(doc: str, sub: str) -> dict:
    """The TyDi-style byte span of `sub` within `doc` (UTF-8 offsets)."""
    data = doc.encode("utf-8")
    start = data.index(sub.encode("utf-8"))
    return {"plaintext_start_byte": start, "plaintext_end_byte": start + len(sub.encode("utf-8"))}


def annotation(passage_index: int, answer: str | None = None, doc: str = "", yes_no: str = "NONE"):
    minimal = byte_span(doc, answer) if answer else {"plaintext_start_byte": -1, "plaintext_end_byte": -1}
    return {
        "annotation_id": 1,
        "passage_answer": {"candidate_index": passage_index},
        "minimal_answer": minimal,
        "yes_no_answer": yes_no,
    }


def make_record(doc, candidates, annotations, title="Makala", question_text="nakala ya asr", example_id=1):
    return {
        "example_id": example_id,
        "language": "swahili",
        "document_title": title,
        "document_url": "https://sw.wikipedia.org/wiki/x",
        "document_plaintext": doc,
        "question_text": question_text,
        "passage_answer_candidates": [byte_span(doc, c) for c in candidates],
        "annotations": annotations,
    }


DOC = "Aya ya sifuri. Jibu ni Nairobi mjini. Aya ya tatu."
CANDIDATES = ["Aya ya sifuri.", "Jibu ni Nairobi mjini.", "Aya ya tatu."]


def test_decode_span_indexes_bytes_not_characters():
    # "é" is two UTF-8 bytes, so char and byte offsets diverge after it.
    doc_bytes = "aé b".encode()
    assert decode_span(doc_bytes, 0, 1) == "a"
    assert decode_span(doc_bytes, 1, 3) == "é"
    assert decode_span(doc_bytes, 4, 5) == "b"


def test_parse_gold_recovers_plurality_passage_and_deduped_answers():
    record = make_record(
        DOC,
        CANDIDATES,
        [
            annotation(1, "Nairobi", DOC),
            annotation(1, "Nairobi", DOC),
            annotation(-1),
        ],
        title="Nairobi",
    )
    gold = parse_gold(record)
    assert gold is not None
    assert gold.title == "Nairobi"
    assert gold.passage_index == 1
    assert gold.passage == "Jibu ni Nairobi mjini."
    assert gold.answers == ("Nairobi",)  # two annotators, one deduped answer
    assert gold.yes_no_answer == "NONE"


def test_parse_gold_is_none_when_no_annotator_chose_a_passage():
    record = make_record(DOC, CANDIDATES, [annotation(-1), annotation(-1)])
    assert parse_gold(record) is None


def test_parse_gold_breaks_ties_to_smallest_index_and_keeps_only_gold_passage_answers():
    # One vote each for passage 0 and 2 -> tie -> smallest (0); the passage-2 answer is dropped.
    record = make_record(
        DOC,
        CANDIDATES,
        [annotation(0, "Aya ya sifuri.", DOC), annotation(2, "Aya ya tatu.", DOC)],
    )
    gold = parse_gold(record)
    assert gold.passage_index == 0
    assert gold.answers == ("Aya ya sifuri.",)


def test_parse_gold_allows_a_passage_with_no_minimal_answer():
    record = make_record(DOC, CANDIDATES, [annotation(1), annotation(1)])
    gold = parse_gold(record)
    assert gold is not None and gold.passage_index == 1
    assert gold.answers == ()


def _transcripts(**by_condition):
    return dict(by_condition)


def test_build_eval_set_keeps_answerable_questions_recorded_in_both_varieties():
    gold_by_id = {
        1: Gold("T1", 0, "Passage 1", ("jibu",), "NONE"),
        2: Gold("T2", 0, "Passage 2", ("jibu",), "NONE"),
        3: None,  # unanswerable
    }
    # The two Google locales are identical per variety (as in the real data).
    transcripts = {
        "kenya-sw-KE": {1: "ke-1", 2: "ke-2", 3: "ke-3"},
        "kenya-sw-TZ": {1: "ke-1", 2: "ke-2", 3: "ke-3"},
        "tanzania-sw-KE": {1: "tz-1", 3: "tz-3"},  # id 2 has no Tanzanian recording
        "tanzania-sw-TZ": {1: "tz-1", 3: "tz-3"},
    }
    gold_questions = {1: "Swali la kwanza?", 2: "Swali la pili?", 3: "Swali la tatu?"}

    items, counts = build_eval_set(gold_by_id, transcripts, gold_questions)

    assert [i.example_id for i in items] == [1]  # only id 1 is both-varieties AND answerable
    item = items[0]
    assert item.gold_question == "Swali la kwanza?"
    assert item.gold_passage == "Passage 1"
    assert item.varieties == ("kenya", "tanzania")
    assert item.google_transcripts == {"kenya": "ke-1", "tanzania": "tz-1"}  # collapsed per variety
    assert counts == {
        "questions_total": 3,
        "recorded_both_varieties": 2,  # ids 1 and 3
        "answerable_has_gold_passage": 2,  # ids 1 and 2
        "eval_set": 1,
        "eval_set_with_minimal_answer": 1,
        "eval_set_missing_gold_question": 0,
        "google_locale_disagreements": 0,
    }


def test_build_eval_set_counts_a_disagreement_between_google_locales():
    gold_by_id = {1: Gold("T1", 0, "Passage 1", ("jibu",), "NONE")}
    transcripts = {
        "kenya-sw-KE": {1: "a"},
        "kenya-sw-TZ": {1: "b"},  # locales differ for Kenya
        "tanzania-sw-KE": {1: "c"},
        "tanzania-sw-TZ": {1: "c"},
    }
    items, counts = build_eval_set(gold_by_id, transcripts, gold_questions={1: "Swali?"})
    assert counts["google_locale_disagreements"] == 1
    assert items[0].google_transcripts["kenya"] == "a"  # first locale (sw-KE) wins


def test_build_eval_set_counts_a_missing_gold_question_but_keeps_the_item():
    gold_by_id = {1: Gold("T1", 0, "Passage 1", (), "NONE")}
    transcripts = {
        "kenya-sw-KE": {1: "k"},
        "kenya-sw-TZ": {1: "k"},
        "tanzania-sw-KE": {1: "t"},
        "tanzania-sw-TZ": {1: "t"},
    }
    items, counts = build_eval_set(gold_by_id, transcripts, gold_questions={})
    assert [i.example_id for i in items] == [1]
    assert items[0].gold_question == ""
    assert counts["eval_set_missing_gold_question"] == 1
    assert counts["eval_set_with_minimal_answer"] == 0  # no minimal answer on this item


def test_snapshot_round_trips_an_sdqa_item(tmp_path):
    item = SDQAItem(
        example_id=42,
        gold_question="Swali?",
        document_title="Makala",
        gold_passage="Aya ya dhahabu.",
        gold_passage_index=1,
        gold_answers=("jibu",),
        yes_no_answer="NONE",
        google_transcripts={"kenya-sw-KE": "nakala"},
        varieties=("kenya", "tanzania"),
    )
    write_snapshot([item], tmp_path, "sdqa_sw_test", {"source": "ffaisal93/SD-QA"})
    manifest = json.loads((tmp_path / "sdqa_sw_test.manifest.json").read_text())
    assert manifest["records"] == 1
    assert manifest["records_per_lang"] == {"sw": 1}

    (loaded,) = [json.loads(line) for line in (tmp_path / "sdqa_sw_test.jsonl").read_text().splitlines()]
    assert loaded["example_id"] == 42
    assert loaded["gold_answers"] == ["jibu"]
    assert loaded["google_transcripts"] == {"kenya-sw-KE": "nakala"}
