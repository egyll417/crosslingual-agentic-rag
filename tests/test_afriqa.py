import pytest

from crag.data.afriqa import (
    build_afriqa_questions,
    parse_answers,
    recover_answers,
    row_to_questions,
)

# The two answer strings in the Swahili test split that ast.literal_eval rejects.
ROW_75_ANSWER = "['I'm Sprung']"
ROW_290_ANSWER_EN = (
    "['bin Laden married Najwa Ghanem at Latakia, Syria; but they were later separated and she "
    "left Afghanistan on September 9, 2001. Bin Laden's other known wives were Khadijah Sharif "
    "(married 1983, divorced 1990s); Khairiah Sabar (married 1985); Siham Sabar (married 1987); "
    "and Amal al-Sadah (married 2000)']"
)


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


def test_both_real_unparseable_answers_recover_as_one_answer():
    assert recover_answers(ROW_75_ANSWER) == ("I'm Sprung",)
    (answer,) = recover_answers(ROW_290_ANSWER_EN)
    assert answer.startswith("bin Laden married Najwa Ghanem")
    assert answer.endswith("and Amal al-Sadah (married 2000)")
    assert len(answer.split()) == 46


def test_parse_answers_falls_back_to_recovery_and_strips():
    assert parse_answers(ROW_75_ANSWER) == ("I'm Sprung",)
    assert parse_answers("[' I'm Sprung ']") == ("I'm Sprung",)


def test_string_that_looks_like_several_answers_is_an_error():
    with pytest.raises(ValueError, match="looks like several answers"):
        recover_answers("['I'm Sprung', 'Bartender']")


def test_string_without_the_list_wrapper_is_an_error():
    with pytest.raises(ValueError, match="unrecognised answer format"):
        recover_answers("I'm Sprung")


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
