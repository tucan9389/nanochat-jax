"""Task base class plus mixture / sequence helpers.

- :class:`Task`: abstract base for a dataset of conversations plus metadata
  and eval criteria. Supports lightweight slicing via ``start/stop/step``.
- :class:`TaskMixture`: multiple tasks combined with a deterministic shuffle
  (seed=42), used to assemble the SFT training mixture.
- :class:`TaskSequence`: sequential concatenation (e.g. curriculum learning).
- :func:`render_mc`: multiple-choice rendering. The letter goes AFTER the
  choice text and no whitespace separates them, which improves binding for
  small models with byte-level BPE.
"""

from __future__ import annotations

import random


class Task:
    """Base class of a Task. Allows for lightweight slicing of the underlying dataset.

    1:1 mirror of upstream ``tasks/common.py:Task``.
    """

    def __init__(self, start: int = 0, stop: int | None = None, step: int = 1) -> None:
        # allows a lightweight logical view over a dataset
        assert start >= 0, f"Start must be non-negative, got {start}"
        assert stop is None or stop >= start, (
            f"Stop should be greater than or equal to start, got {stop} and {start}"
        )
        assert step >= 1, f"Step must be strictly positive, got {step}"
        self.start = start
        self.stop = stop  # could be None here
        self.step = step

    @property
    def eval_type(self) -> str:
        """One of 'generative' | 'categorical'."""
        raise NotImplementedError

    def num_examples(self) -> int:
        raise NotImplementedError

    def get_example(self, index: int) -> dict:
        raise NotImplementedError

    def __len__(self) -> int:
        start = self.start
        stop = self.num_examples() if self.stop is None else self.stop
        step = self.step
        span = stop - start
        num = (span + step - 1) // step  # ceil_div(span, step)
        assert num >= 0, f"Negative number of examples???: {num}"  # prevent footguns
        return num

    def __getitem__(self, index: int) -> dict:
        assert isinstance(index, int), f"Index must be an integer, got {type(index)}"
        physical_index = self.start + index * self.step
        conversation = self.get_example(physical_index)
        return conversation

    def evaluate(self, problem: dict, completion: str) -> int | bool:
        raise NotImplementedError


class TaskMixture(Task):
    """For SFT Training it becomes useful to train on a mixture of datasets.

    1:1 mirror of upstream ``tasks/common.py:TaskMixture``.

    Fun trick: if you wish to oversample any task, just pass it in multiple times in the list.
    Deterministic shuffle (seed=42) ensures tasks are mixed throughout training, regardless of
    dataset size.
    """

    def __init__(self, tasks: list[Task], **kwargs) -> None:
        super().__init__(**kwargs)
        # tasks is a list of Task objects
        self.tasks = tasks
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum(self.lengths)
        # Build list of all (task_idx, local_idx) pairs
        self.index_map: list[tuple[int, int]] = []
        for task_idx, task_length in enumerate(self.lengths):
            for local_idx in range(task_length):
                self.index_map.append((task_idx, local_idx))
        # Deterministically shuffle to mix tasks throughout training
        rng = random.Random(42)
        rng.shuffle(self.index_map)
        # Note: this is not the most elegant or best solution, but it's ok for now

    def num_examples(self) -> int:
        return self.num_conversations

    def get_example(self, index: int) -> dict:
        """Access conversations according to a deterministic shuffle of all examples.

        This ensures tasks are mixed throughout training, regardless of dataset size.
        """
        assert 0 <= index < self.num_conversations, (
            f"Index {index} out of range for mixture with "
            f"{self.num_conversations} conversations"
        )
        task_idx, local_idx = self.index_map[index]
        return self.tasks[task_idx][local_idx]


class TaskSequence(Task):
    """For SFT Training sometimes we want to sequentially train on a list of tasks.

    1:1 mirror of upstream ``tasks/common.py:TaskSequence``.

    Useful for cases that require a training curriculum.
    """

    def __init__(self, tasks: list[Task], **kwargs) -> None:
        super().__init__(**kwargs)
        self.tasks = tasks
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum(self.lengths)

    def num_examples(self) -> int:
        return self.num_conversations

    def get_example(self, index: int) -> dict:
        assert 0 <= index < self.num_conversations, (
            f"Index {index} out of range for sequence with "
            f"{self.num_conversations} conversations"
        )
        for task_idx, task_length in enumerate(self.lengths):
            if index < task_length:
                return self.tasks[task_idx][index]
            index -= task_length
        # unreachable due to assert above, but keeps type checkers happy
        raise AssertionError(f"Index {index} not found in any task")


def render_mc(question: str, letters: tuple[str, ...], choices: list[str]) -> str:
    """The common multiple choice rendering format we will use.

    1:1 mirror of upstream ``tasks/common.py:render_mc``.

    Note two important design decisions:

    1) Bigger models don't care as much, but smaller models prefer to have
       the letter *after* the choice, which results in better binding.
    2) There is no whitespace between the delimiter (=) and the letter.
       This is actually critical because the tokenizer has different token ids
       for " A" vs. "A". The assistant responses will be just the letter itself,
       i.e. "A", so it is important that here in the prompt it is the exact same
       token, i.e. "A" with no whitespace before it. Again, bigger models don't care
       about this too much, but smaller models do care about some of these details.
    """
    query = f"Multiple Choice question: {question}\n"
    query += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters, choices, strict=True)])
    query += "\nRespond only with the letter of the correct answer."
    return query
