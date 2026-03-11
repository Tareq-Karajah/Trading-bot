# Strategy v3.0 Specification

## Overview
Hybrid strategy combining trend following and mean reversion.

## Components

### 1. Trend Filter
- **Indicator**: EMA (50) vs EMA (200)
- **Logic**: Long if EMA50 > EMA200, Short if EMA50 < EMA200

### 2. Entry Trigger
- **Indicator**: RSI (14)
- **Logic**:
  - Long: RSI < 30 (Oversold) in Uptrend
  - Short: RSI > 70 (Overbought) in Downtrend

### 3. Exit Logic
- **Take Profit**: 2 * ATR (14)
- **Stop Loss**: 1 * ATR (14)
- **Time Exit**: Close after N bars if no target hit.
