from collections import deque
from typing import Deque, Optional, List
from trading_bot.core.models import OHLCV

class OHLCVBuffer:
    """
    In-memory rolling buffer for OHLCV data.
    Maintains a fixed size history for indicator calculation.
    """
    
    def __init__(self, size: int = 200):
        self.size = size
        self.buffer: Deque[OHLCV] = deque(maxlen=size)

    def add(self, candle: OHLCV) -> None:
        """
        Add a new candle to the buffer.
        """
        self.buffer.append(candle)

    def get_all(self) -> List[OHLCV]:
        """
        Return all candles in the buffer as a list.
        """
        return list(self.buffer)

    def clear(self) -> None:
        """
        Clear the buffer.
        """
        self.buffer.clear()
    
    @property
    def is_full(self) -> bool:
        return len(self.buffer) == self.size
