from typing import Any

from trulens_eval import Provider


class Compare(Provider):

    def equals(self, actual: Any, expected: Any) -> float:  # noqa
        return 1.0 if actual == expected else 0.0
