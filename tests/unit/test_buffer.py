from datetime import datetime
import pytest
from trading_bot.data.buffer import OHLCVBuffer
from trading_bot.core.models import OHLCV

def test_buffer_initialization():
    buffer = OHLCVBuffer(size=10)
    assert buffer.size == 10
    assert len(buffer.buffer) == 0
    assert not buffer.is_full

def test_buffer_add_and_overflow():
    buffer = OHLCVBuffer(size=3)
    
    # Create dummy candles
    c1 = OHLCV(timestamp=datetime.now(), open=100, high=105, low=95, close=102, volume=1000, symbol="BTC/USD", timeframe="1m")
    c2 = OHLCV(timestamp=datetime.now(), open=102, high=107, low=100, close=105, volume=1200, symbol="BTC/USD", timeframe="1m")
    c3 = OHLCV(timestamp=datetime.now(), open=105, high=110, low=102, close=108, volume=1500, symbol="BTC/USD", timeframe="1m")
    c4 = OHLCV(timestamp=datetime.now(), open=108, high=112, low=105, close=110, volume=1100, symbol="BTC/USD", timeframe="1m")
    
    buffer.add(c1)
    assert len(buffer.buffer) == 1
    assert buffer.get_all() == [c1]
    
    buffer.add(c2)
    buffer.add(c3)
    assert len(buffer.buffer) == 3
    assert buffer.is_full
    assert buffer.get_all() == [c1, c2, c3]
    
    # Adding 4th element should remove the first one
    buffer.add(c4)
    assert len(buffer.buffer) == 3
    assert buffer.get_all() == [c2, c3, c4]

def test_buffer_clear():
    buffer = OHLCVBuffer(size=5)
    c1 = OHLCV(timestamp=datetime.now(), open=100, high=105, low=95, close=102, volume=1000, symbol="BTC/USD", timeframe="1m")
    buffer.add(c1)
    
    assert len(buffer.buffer) == 1
    buffer.clear()
    assert len(buffer.buffer) == 0
    assert not buffer.is_full
