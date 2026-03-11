# Event Contracts

This document defines the contract for events within the trading bot system.

## Event Types

### MarketDataEvent
- **Source**: Data Ingestion
- **Payload**: `OHLCV` or `Trade` object
- **Description**: Represents a new market data point.

### SignalEvent
- **Source**: Strategy Engine
- **Payload**: `Signal` object
- **Description**: Generated when a strategy condition is met.

### OrderEvent
- **Source**: Risk Manager / Execution Engine
- **Payload**: `Order` object
- **Description**: Represents an order to be sent to the exchange.

### FillEvent
- **Source**: Exchange Adapter
- **Payload**: `Fill` object
- **Description**: Confirmation of an order execution.
