"""
Centralized configuration for special tokens used in the model.
"""

from typing import List

class TokenConfig:
    """Defines all special tokens for the tokenizer."""

    # Core structural tokens
    GRAPH_START = "<graph_start>"
    GRAPH_END = "<graph_end>"
    NODE = "<node>"

    # Day and time tokens for time-aware querying
    DAY_PREFIX = "<DAY_"
    TIME_PREFIX = "<TIME_"
    DAY_END = ">"
    TIME_END = ">"

    # Define ranges
    MAX_DAYS = 100  # e.g., days 0-99
    MAX_HOURS = 24  # e.g., hours 0-23

    @classmethod
    def get_day_token(cls, day: int) -> str:
        """Returns the special token for a given day."""
        if 0 <= day < cls.MAX_DAYS:
            return f"{cls.DAY_PREFIX}{day}{cls.DAY_END}"
        raise ValueError(f"Day must be between 0 and {cls.MAX_DAYS - 1}")

    @classmethod
    def get_time_token(cls, hour: int) -> str:
        """Returns the special token for a given hour."""
        if 0 <= hour < cls.MAX_HOURS:
            return f"{cls.TIME_PREFIX}{hour}{cls.TIME_END}"
        raise ValueError(f"Hour must be between 0 and {cls.MAX_HOURS - 1}")

    @classmethod
    def get_all_special_tokens(cls) -> List[str]:
        """Returns a list of all special tokens to be added to the tokenizer."""
        tokens = [cls.GRAPH_START, cls.GRAPH_END, cls.NODE]
        
        day_tokens = [cls.get_day_token(i) for i in range(cls.MAX_DAYS)]
        time_tokens = [cls.get_time_token(i) for i in range(cls.MAX_HOURS)]
        
        tokens.extend(day_tokens)
        tokens.extend(time_tokens)
        
        return tokens 