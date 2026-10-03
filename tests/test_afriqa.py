import pytest

from crag.data.afriqa import build_afriqa_questions, parse_answers, row_to_questions


def make_row(question="Je,papa wa roma wa kwanza aliitwa nani?", answers="['Mt Petro']"):
    return {
        "question": question,
        "answers": answers,
        "lang": "swa",
        "split": "test",
        "translated_question": "What was the first pope of Rome called?",
        "translated_answer": "['St Peter']",
        "translation_type": "human_translation",
    }


def test_row_gives_swahili_and_english_records_with_shared_parallel_id():
    sw, en = row_to_questions(make_row(), "afriqa-swa-test-0002")
    assert (sw.lang, en.lang) == ("sw", "en")
    assert (sw.id, en.id) == ("afriqa-swa-test-0002-sw", "afriqa-swa-test-0002-en")
    assert sw.parallel_id == en.parallel_id == "afriqa-swa-test-0002"
    assert sw.text == "Je,papa wa roma wa kwanza aliitwa nani?"
    assert en.text == "What was the first pope of Rome called?"


def test_both_records_carry_the_english_gold_answer():
    sw, en = row_to_questions(make_row(), "afriqa-swa-test-0002")
    assert sw.answers == ("Mt Petro",)
    assert en.answers == ("St Peter",)
    assert sw.answers_en == en.answers_en == ("St Peter",)
    assert sw.answer_type is None and en.answer_type is None
    assert sw.source == en.source == "afriqa"


def test_parse_answers_strips_and_dedupes():
    assert parse_answers("['Webuye']") == ("Webuye",)
    assert parse_answers("[' Oginga Odinga ', 'Odinga', 'Oginga Odinga', '']") == (
        "Oginga Odinga",
        "Odinga",
    )


def test_row_without_an_english_gold_answer_is_an_error():
    row = make_row() | {"translated_answer": "[]"}
    with pytest.raises(ValueError, match="no English gold answer"):
        row_to_questions(row, "afriqa-swa-test-0000")


def test_build_ids_follow_row_position():
    rows = [make_row(question="swali la kwanza"), make_row(question="swali la pili")]
    questions = build_afriqa_questions(rows)
    assert [q.id for q in questions] == [
        "afriqa-swa-test-0000-sw",
        "afriqa-swa-test-0000-en",
        "afriqa-swa-test-0001-sw",
        "afriqa-swa-test-0001-en",
    ]
