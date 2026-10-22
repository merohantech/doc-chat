import math


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 characters per token for English prose).

    Only used for limits and display before a request is made; the provider's
    reported `prompt_tokens` is the real number. Errs slightly high for prose,
    which is the safe direction for a limit.
    """
    return math.ceil(len(text) / 4)
