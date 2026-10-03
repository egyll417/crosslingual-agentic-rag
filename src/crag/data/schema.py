from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Question:
    """One question in one language. Language versions of a question share parallel_id."""

    id: str
    parallel_id: str
    lang: str
    text: str
    answers: tuple[str, ...]  # gold answers and aliases in the question's language
    answers_en: tuple[str, ...]  # gold answers and aliases in English, the knowledge-base language
    answer_type: str | None
    source: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Question":
        return cls(
            **{**data, "answers": tuple(data["answers"]), "answers_en": tuple(data["answers_en"])}
        )
