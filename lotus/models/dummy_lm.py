import random
from typing import Any

from lotus.models.lm import LM
from lotus.types import LMOutput

class DummyLM(LM):
    """
    A dummy language model for testing sem_filter and sem_topk without
    making real API calls.

    - sem_filter: randomly returns "True" or "False" for each input
    - sem_topk: randomly picks "Document 1" or "Document 2" for each
      pairwise comparison

    For any other operation, returns an empty string.

    Usage:
        import lotus
        from lotus.models import DummyLM

        lm = DummyLM()
        lotus.settings.configure(lm=lm)

        df.sem_filter("The {text} is positive")
        df.sem_topk("The most relevant {title}", K=3)
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs["model"] = "dummy"
        super().__init__(**kwargs)

    @staticmethod
    def _is_filter_prompt(messages: list[dict[str, Any]]) -> bool:
        """
        Checks if the message is a filter message by checking if the phrase
        'true for the given context' exists in the content. Filter messages always contain this text, 
        from filter_formatter in task_instructions.py.
        """
        if not messages:
            return False
        first = messages[0]
        if first.get("role") != "system":
            return False
        content = first.get("content", "")
        return "true for the given context" in content.lower()

    @staticmethod
    def _is_topk_prompt(messages: list[dict[str, Any]]) -> bool:
        """
        Checks if the message is a filter message by checking if the phrase
        'select and return the most relevant document' exists in the content. Filter 
        messages always contain this text from get_match_prompt_binary in sem_topk.py.
        """
        if not messages:
            return False
        first = messages[0]
        if first.get("role") != "system":
            return False
        content = first.get("content", "")
        return "select and return the most relevant document" in content.lower()

    def __call__(
        self,
        messages: list[list[dict[str, Any]]],
        show_progress_bar: bool = False,
        progress_bar_desc: str = "",
        **kwargs: Any,
    ) -> LMOutput:
        """
        Return random responses without any API call, corrected for type of operation.
        """
        outputs: list[str] = []

        for prompt_messages in messages:
            if self._is_filter_prompt(prompt_messages):
                # If operation is type filter, return true/false randomly
                outputs.append(random.choice(["True", "False"]))

            elif self._is_topk_prompt(prompt_messages):
                # If operation is type topk, pick a random document
                outputs.append(random.choice(["Document 1", "Document 2"]))

            else:
                # Unsupported semantic operation
                outputs.append("")

        return LMOutput(outputs=outputs, logprobs=None)

    def count_tokens(self, messages: list[dict[str, Any]] | str) -> int:
        return 0

    def encode_text(self, text: str) -> list[int]:
        return []

    def decode_tokens(self, tokens: list[int]) -> str:
        return ""

    def print_total_usage(self) -> None:
        print("DummyLM: no real API calls were made.")

    def get_model_name(self) -> str:
        return "dummy"

    def is_deepseek(self) -> bool:
        return False