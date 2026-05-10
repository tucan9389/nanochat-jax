"""SmolTalk task: HuggingFace ``HuggingFaceTB/smol-smoltalk`` conversations.

The "smol" version is appropriate for smaller models. Used as the general
conversation anchor in the SFT mixture.

- ``train`` split: 460,341 rows of multi-turn conversations
- ``test`` split: 24,229 rows (used for val_dataset)

Each row is a conversation with an optional system message followed by
alternating user / assistant messages.
"""

from __future__ import annotations

from datasets import load_dataset

from nanochat_jax.tasks.common import Task


class SmolTalk(Task):
    """ smol-smoltalk dataset. train is 460K rows, test is 24K rows.

    PT 1:1 mirror: ``nanochat/tasks/smoltalk.py:SmolTalk``.
    """

    def __init__(self, split: str, **kwargs) -> None:
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SmolTalk split must be train|test"
        self.ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self) -> int:
        return self.length

    def get_example(self, index: int) -> dict:
        row = self.ds[index]
        messages = row["messages"]
        # ---------------------------------------------------------------------
        # sanity checking asserts here (PT 1:1 — protect against footguns)
        # there is an optional system message at the beginning
        assert len(messages) >= 1
        first_message = messages[0]
        if first_message["role"] == "system":
            rest_messages = messages[1:]  # optional system message is OK
        else:
            rest_messages = messages
        assert len(rest_messages) >= 2, (
            "SmolTalk messages must have at least 2 messages"
        )
        for i, message in enumerate(rest_messages):
            # user and assistant alternate as user, assistant, user, assistant, ...
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected_role, (
                f"Message {i} has role {message['role']} but should be {expected_role}"
            )
            assert isinstance(message["content"], str), "Content must be a string"
        # ---------------------------------------------------------------------
        # create and return the Conversation object (ok to emit the system message too)
        conversation = {
            "messages": messages,
        }
        return conversation
