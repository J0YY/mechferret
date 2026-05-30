"""Canonical interpretability tasks.

Each task is a small, well-understood behavioural probe used throughout the
mechanistic-interpretability literature. A task carries clean prompts, a
matched corrupted variant (for activation patching), and the answer tokens
whose logit difference is the standard metric.

These are intentionally lightweight: they exist so the agent can reason about
*which* behaviour it is investigating and so both the synthetic and real
backends share the same task definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Task:
    name: str
    description: str
    metric: str
    clean_prompts: tuple[str, ...]
    corrupt_prompts: tuple[str, ...]
    answers: tuple[tuple[str, str], ...]  # (correct, incorrect) token pairs
    reference: str = ""
    expected_components: tuple[str, ...] = field(default_factory=tuple)


_IOI_NAMES = [
    ("John", "Mary"),
    ("Alice", "Bob"),
    ("Tom", "Sarah"),
    ("David", "Anna"),
]


def _ioi() -> Task:
    clean = tuple(
        f"When {a} and {b} went to the store, {b} gave a drink to" for a, b in _IOI_NAMES
    )
    corrupt = tuple(
        f"When {a} and {b} went to the store, {c} gave a drink to"
        for (a, b), (c, _) in zip(_IOI_NAMES, [("Carl", ""), ("Dana", ""), ("Eve", ""), ("Finn", "")])
    )
    answers = tuple((a, b) for a, b in _IOI_NAMES)
    return Task(
        name="ioi",
        description="Indirect Object Identification: predict the indirect object name.",
        metric="logit_diff",
        clean_prompts=clean,
        corrupt_prompts=corrupt,
        answers=answers,
        reference="Wang et al. 2022, 'Interpretability in the Wild' (IOI circuit).",
        expected_components=("name_mover_head", "s_inhibition_head", "duplicate_token_head"),
    )


def _induction() -> Task:
    clean = (
        "A B C D E A B C D",
        "the cat sat the cat",
        "1 2 3 4 5 1 2 3 4",
        "red blue green red blue",
    )
    corrupt = (
        "A B C D E F G H I",
        "the dog ran a bird flew",
        "1 2 3 4 5 6 7 8 9",
        "red blue green yellow pink",
    )
    answers = (("E", "F"), ("sat", "ran"), ("5", "6"), ("green", "yellow"))
    return Task(
        name="induction",
        description="Induction: continue a repeated sequence by copying what followed last time.",
        metric="logit_diff",
        clean_prompts=clean,
        corrupt_prompts=corrupt,
        answers=answers,
        reference="Olsson et al. 2022, 'In-context Learning and Induction Heads'.",
        expected_components=("induction_head", "previous_token_head"),
    )


def _greater_than() -> Task:
    clean = (
        "The war lasted from the year 1732 to the year 17",
        "The event ran from 1845 to 18",
        "Active between 1903 and 19",
        "Recorded from 1660 to 16",
    )
    corrupt = (
        "The war lasted from the year 1700 to the year 17",
        "The event ran from 1800 to 18",
        "Active between 1900 and 19",
        "Recorded from 1600 to 16",
    )
    answers = (("33", "31"), ("46", "44"), ("04", "02"), ("61", "59"))
    return Task(
        name="greater_than",
        description="Greater-than: predict a year strictly greater than the start year.",
        metric="prob_diff",
        clean_prompts=clean,
        corrupt_prompts=corrupt,
        answers=answers,
        reference="Hanna et al. 2023, 'How does GPT-2 compute greater-than?'.",
        expected_components=("mlp_comparison", "attention_to_year"),
    )


def _factual_recall() -> Task:
    clean = (
        "The Eiffel Tower is located in the city of",
        "The capital of Japan is",
        "Water is made of hydrogen and",
        "The author of Romeo and Juliet is",
    )
    corrupt = (
        "The Colosseum is located in the city of",
        "The capital of France is",
        "Salt is made of sodium and",
        "The author of Hamlet is",
    )
    answers = (("Paris", "Rome"), ("Tokyo", "Paris"), ("oxygen", "chlorine"), ("Shakespeare", "Dickens"))
    return Task(
        name="factual_recall",
        description="Factual recall: retrieve a stored fact (ROME-style causal tracing target).",
        metric="logit_diff",
        clean_prompts=clean,
        corrupt_prompts=corrupt,
        answers=answers,
        reference="Meng et al. 2022, 'Locating and Editing Factual Associations' (ROME).",
        expected_components=("mlp_factual_store", "subject_attention_head"),
    )


TASKS: dict[str, Task] = {
    task.name: task
    for task in (_ioi(), _induction(), _greater_than(), _factual_recall())
}


def get_task(name: str) -> Task:
    key = (name or "").strip().lower()
    if key not in TASKS:
        raise KeyError(f"Unknown interpretability task: {name!r}. Known: {sorted(TASKS)}")
    return TASKS[key]


def infer_task(question: str) -> str:
    """Heuristically pick the most relevant task from a free-text question."""

    text = (question or "").lower()
    keyword_map = {
        "ioi": ("ioi", "indirect object", "name mover", "name-mover"),
        "induction": ("induction", "in-context", "copy", "repeated"),
        "greater_than": ("greater", "greater-than", "year", "comparison", "numeric"),
        "factual_recall": ("fact", "factual", "recall", "knowledge", "rome", "association"),
    }
    for task_name, keywords in keyword_map.items():
        if any(keyword in text for keyword in keywords):
            return task_name
    return "ioi"
