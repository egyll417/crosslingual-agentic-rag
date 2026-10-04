import pytest

from crag.eval.metrics import first_hit_rank, has_answer, normalize, recall_at_k


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello, World!", "hello world"),
        # Punctuation becomes a space, so words on either side stay separate words.
        ("Smith's", "smith s"),
        ("2016–2017", "2016 2017"),
        ("U.S.", "u s"),
        # Unicode punctuation, not just ASCII.
        ("it’s", "it s"),
        ("«Bonjour»", "bonjour"),
        ("„Guten Tag“", "guten tag"),
        # Tabs, newlines and non-breaking spaces collapse; both ends are trimmed.
        (" a\tb\nc d  ", "a b c d"),
        # Diacritics are kept.
        ("Zürich", "zürich"),
        # NFKC folds compatibility characters such as fullwidth digits.
        ("２０１６", "2016"),
        ("", ""),
        ("!!!", ""),
    ],
)
def test_normalize(text, expected):
    assert normalize(text) == expected


def test_normalize_composed_and_decomposed_accents_are_equal():
    assert normalize("Zürich") == normalize("Zürich")


@pytest.mark.parametrize("text", ["Hello, World!", "Smith's 2016–2017", " a\tb  ", "Zürich"])
def test_normalize_is_idempotent(text):
    assert normalize(normalize(text)) == normalize(text)


@pytest.mark.parametrize(
    ("answer", "passage"),
    [
        # The trap: a number inside a longer number is not a match.
        ("2", "He won in 2016."),
        ("2", "the 12 apostles"),
        ("art", "a smart artist"),
        ("New York", "a New Yorker cartoon"),
        # A multi-word answer must appear as a contiguous run, in order.
        ("New York", "from New Jersey to York"),
        ("York New", "born in New York"),
    ],
)
def test_has_answer_rejects_partial_words(answer, passage):
    assert has_answer(passage, [answer]) is False


@pytest.mark.parametrize(
    ("answer", "passage"),
    [
        ("2", "2 titles"),
        ("2", "titles: 2."),
        ("2", "He won 2 titles in 2016."),
        ("New York", "born in New York City"),
        ("Smith", "Smith's book"),
        ("2017", "the 2016–2017 season"),
        ("Barack Obama", "BARACK OBAMA, president"),
    ],
)
def test_has_answer_matches_whole_words(answer, passage):
    assert has_answer(passage, [answer]) is True


def test_has_answer_any_alias_is_enough():
    assert has_answer("Mumbai is a city.", ("Bombay", "Mumbai")) is True
    assert has_answer("Delhi is a city.", ("Bombay", "Mumbai")) is False


def test_has_answer_no_answers_is_false():
    assert has_answer("Mumbai is a city.", []) is False


@pytest.mark.parametrize("alias", ["", "!!", " "])
def test_has_answer_empty_alias_never_matches(alias):
    # `"" in passage` is always True; an empty alias must not make every passage a hit.
    assert has_answer("Mumbai is a city.", [alias]) is False


def test_has_answer_empty_alias_does_not_hide_other_aliases():
    assert has_answer("Mumbai is a city.", ["", "Mumbai"]) is True


@pytest.mark.parametrize(
    ("passages", "expected"),
    [
        # The trap passage comes first and must be skipped.
        (["Founded in 2016.", "It has 2 floors."], 2),
        (["It has 2 floors.", "Founded in 2016."], 1),
        (["Founded in 2016."], None),
        ([], None),
    ],
)
def test_first_hit_rank(passages, expected):
    assert first_hit_rank(passages, ["2"]) == expected


def test_first_hit_rank_rejects_a_bare_string_as_answers():
    # A str is iterable, so "Mumbai" would otherwise be matched letter by letter.
    with pytest.raises(TypeError):
        first_hit_rank(["Mumbai is a city."], "Mumbai")


@pytest.mark.parametrize(("k", "expected"), [(1, 0.25), (2, 0.5), (3, 0.75), (100, 0.75)])
def test_recall_at_k(k, expected):
    assert recall_at_k([1, 3, None, 2], k) == expected


def test_recall_at_k_no_hits_is_zero():
    assert recall_at_k([None, None], 5) == 0.0


@pytest.mark.parametrize("k", [0, -1])
def test_recall_at_k_rejects_k_below_one(k):
    with pytest.raises(ValueError):
        recall_at_k([1, 2], k)


def test_recall_at_k_rejects_empty_ranks():
    with pytest.raises(ValueError):
        recall_at_k([], 5)
