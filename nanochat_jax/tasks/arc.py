"""
The ARC dataset from Allen AI.
https://huggingface.co/datasets/allenai/ai2_arc

PT 1:1 mirror — substitution 0 (algorithm framework-agnostic, datasets lib only).
"""

from datasets import load_dataset

from .common import Task, render_mc


class ARC(Task):

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["ARC-Easy", "ARC-Challenge"], "ARC subset must be ARC-Easy or ARC-Challenge"
        assert split in ["train", "validation", "test"], "ARC split must be train|validation|test"
        self.ds = load_dataset("allenai/ai2_arc", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return "categorical"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"]  # the question text
        choices = row["choices"]["text"]  # the text of each choice
        answer_string = row["answerKey"]  # e.g. "A", "B", "C", "D"
        letters = row["choices"]["label"]  # e.g. ["A", "B", "C", "D"]
        assert answer_string in letters, f"ARC answer {answer_string} must be one of {letters}"
        # create and return the Conversation object
        user_message = render_mc(question, letters, choices)
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer_string},
        ]
        conversation = {
            "messages": messages,
            "letters": letters,  # useful during evaluation, narrow predicted letter to one of letters
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        # the assert prevents footguns; matches PT 1:1
        assert assistant_response in conversation["letters"], (
            f"ARC answer {assistant_response} is expected to be one of {conversation['letters']}"
        )
        assistant_message = conversation["messages"][-1]["content"]  # e.g. "A"
        return assistant_response == assistant_message
