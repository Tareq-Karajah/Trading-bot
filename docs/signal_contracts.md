# Signal Contracts

Standardized format for trading signals.

## Signal Structure

```python
class Signal:
    timestamp: datetime
    symbol: str
    direction: Direction (LONG/SHORT)
    strategy_id: str
    confidence: float (0.0 - 1.0)
    metadata: Dict
```

## Signal Lifecycle
1. **Generated**: Created by Strategy Engine.
2. **Validated**: Checked by Risk Manager.
3. **Executed**: Converted to Order by Execution Engine.
4. **Expired**: Discarded if not acted upon within TTL.
