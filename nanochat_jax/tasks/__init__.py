"""Evaluation and SFT tasks.

Modules:
- :mod:`tasks.common` -- :class:`Task` base + :class:`TaskMixture` + :class:`TaskSequence` + :func:`render_mc`
- :mod:`tasks.customjson` -- :class:`CustomJSON` (JSONL conversation loader)
- :mod:`tasks.smoltalk` -- :class:`SmolTalk` (HuggingFace ``HuggingFaceTB/smol-smoltalk``)
- :mod:`tasks.arc` -- ``ARC-Easy`` / ``ARC-Challenge``
- :mod:`tasks.mmlu` -- ``MMLU``
- :mod:`tasks.gsm8k` -- ``GSM8K``
- :mod:`tasks.humaneval` -- ``HumanEval`` (with sandboxed execution)
- :mod:`tasks.spellingbee` -- ``SpellingBee`` / ``SimpleSpelling``

Tasks that pull from the ``datasets`` library are NOT imported eagerly here
(``from datasets import load_dataset`` is slow at startup). Callers should
import from the explicit module path, e.g.
``from nanochat_jax.tasks.arc import ARC``.
"""

from nanochat_jax.tasks.common import Task, TaskMixture, TaskSequence, render_mc
from nanochat_jax.tasks.customjson import CustomJSON
from nanochat_jax.tasks.smoltalk import SmolTalk

__all__ = [
    "CustomJSON",
    "SmolTalk",
    "Task",
    "TaskMixture",
    "TaskSequence",
    "render_mc",
]
