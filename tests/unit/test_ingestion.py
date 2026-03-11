import pytest
from trading_bot.data.ingestion import DataIngestion

@pytest.mark.asyncio
async def test_ingestion_stub():
    ingestion = DataIngestion()
    
    # Test connect (stub)
    await ingestion.connect()
    
    # Test subscribe (stub)
    await ingestion.subscribe(["BTC/USD"])
    
    # Test get_latest_candle (stub)
    candle = await ingestion.get_latest_candle("BTC/USD")
    assert candle is None
