import json
from pathlib import Path

import pytest

from crag.data.io import read_snapshot, write_snapshot
from crag.data.mkqa import (
    ANSWER_TYPES,
    ELIGIBLE_ANSWER_TYPES,
    answer_strings,
    build_mkqa_questions,
    is_eligible,
    read_sample_ids,
    row_to_questions,
    select_parallel_sample,
    write_sample_ids,
)

TRACKED_SAMPLE_IDS = Path(__file__).parents[1] / "configs" / "mkqa_sample_ids.txt"


def make_answer(answer_type="entity", text="Chris Young", aliases=("Christopher Young",)):
    return {
        "type": ANSWER_TYPES.index(answer_type),
        "entity": "",
        "text": text,
        "aliases": list(aliases),
    }


def make_row(example_id="42", english=None):
    english = english or [make_answer()]
    return {
        "example_id": example_id,
        "query": "who sings it",
        "queries": {
            "en": "who sings it",
            "de": "wer singt es",
            "fr": "qui le chante",
            "it": "chi lo canta",
        },
        "answers": {
            "en": english,
            "de": [make_answer(text="Chris Young", aliases=("Christopher Alan Young",))],
            "fr": [make_answer(text="Chris Young", aliases=())],
        },
    }


def test_row_gives_one_record_per_language_with_shared_parallel_id():
    questions = row_to_questions(make_row())
    assert [q.lang for q in questions] == ["en", "de", "fr"]
    assert {q.parallel_id for q in questions} == {"mkqa-42"}
    assert [q.id for q in questions] == ["mkqa-42-en", "mkqa-42-de", "mkqa-42-fr"]
    assert [q.text for q in questions] == ["who sings it", "wer singt es", "qui le chante"]


def test_every_record_carries_english_aliases_and_answer_type():
    questions = row_to_questions(make_row())
    assert {q.answers_en for q in questions} == {("Chris Young", "Christopher Young")}
    assert questions[1].answers == ("Chris Young", "Christopher Alan Young")
    assert {q.answer_type for q in questions} == {"entity"}
    assert {q.source for q in questions} == {"mkqa"}


def test_answer_strings_dedupes_and_drops_empty():
    answers = [
        make_answer(text="2.0", aliases=("2", "2.0", "")),
        {"type": 4, "entity": "", "text": None, "aliases": None},
        make_answer(text=" two ", aliases=()),
    ]
    assert answer_strings(answers) == ("2.0", "2", "two")


@pytest.mark.parametrize("answer_type", ANSWER_TYPES)
def test_eligibility_follows_answer_type(answer_type):
    row = make_row(english=[make_answer(answer_type=answer_type)])
    assert is_eligible(row) == (answer_type in ELIGIBLE_ANSWER_TYPES)


def test_row_with_any_ineligible_answer_type_is_excluded():
    row = make_row(english=[make_answer("entity"), make_answer("binary", text="yes", aliases=())])
    assert not is_eligible(row)


def test_row_without_an_english_answer_string_is_excluded():
    assert not is_eligible(make_row(english=[make_answer(text="", aliases=())]))


def test_sample_is_fixed_by_seed_and_changes_with_it():
    rows = [make_row(example_id=str(i)) for i in range(50)]
    first = select_parallel_sample(rows, 10, seed=0)
    assert select_parallel_sample(rows, 10, seed=0) == first
    assert select_parallel_sample(rows, 10, seed=1) != first
    assert len({row["example_id"] for row in first}) == 10


def test_sample_does_not_depend_on_row_order():
    rows = [make_row(example_id=str(i)) for i in range(50)]
    assert select_parallel_sample(rows[::-1], 10, seed=0) == select_parallel_sample(
        rows, 10, seed=0
    )


def test_sample_larger_than_eligible_rows_is_an_error():
    rows = [make_row(example_id=str(i)) for i in range(3)]
    with pytest.raises(ValueError, match="only 3 eligible"):
        select_parallel_sample(rows, 4, seed=0)


def test_tracked_sample_ids_file_has_500_unique_ids():
    ids = read_sample_ids(TRACKED_SAMPLE_IDS)
    assert len(ids) == len(set(ids)) == 500


def test_build_selects_the_listed_rows_in_file_order():
    rows = [make_row(example_id=str(i)) for i in range(5)]
    questions = build_mkqa_questions(rows, ["3", "1"])
    assert [q.id for q in questions] == [
        "mkqa-3-en",
        "mkqa-3-de",
        "mkqa-3-fr",
        "mkqa-1-en",
        "mkqa-1-de",
        "mkqa-1-fr",
    ]


def test_sample_id_missing_from_the_dataset_is_an_error():
    rows = [make_row(example_id=str(i)) for i in range(3)]
    with pytest.raises(ValueError, match="1 sample ids are not in the dataset"):
        build_mkqa_questions(rows, ["1", "7"])


def test_sample_id_that_is_no_longer_eligible_is_an_error():
    binary = [make_answer("binary", text="yes", aliases=())]
    rows = [make_row(example_id="1"), make_row(example_id="2", english=binary)]
    with pytest.raises(ValueError, match="1 sample ids are no longer eligible"):
        build_mkqa_questions(rows, ["1", "2"])


def test_duplicate_or_missing_ids_in_the_file_are_an_error(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text("1\n2\n1\n")
    with pytest.raises(ValueError, match="more than once"):
        read_sample_ids(path)
    path.write_text("\n")
    with pytest.raises(ValueError, match="lists no ids"):
        read_sample_ids(path)


def test_resampling_writes_sorted_ids_that_the_loader_reads_back(tmp_path):
    rows = [make_row(example_id=str(i)) for i in range(20)]
    path = tmp_path / "ids.txt"
    ids = write_sample_ids(rows, path, n=5, seed=0)
    assert len(ids) == 5 and ids == sorted(ids)
    assert read_sample_ids(path) == ids


def test_snapshot_round_trips_and_writes_manifest(tmp_path):
    questions = row_to_questions(make_row())
    path = write_snapshot(questions, tmp_path, "mkqa", {"source": "fixture"})
    assert read_snapshot(path) == questions
    manifest = json.loads((tmp_path / "mkqa.manifest.json").read_text())
    assert manifest["source"] == "fixture"
    assert manifest["records"] == 3
    assert manifest["records_per_lang"] == {"de": 1, "en": 1, "fr": 1}
