from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    """
    Application configuration using Pydantic BaseSettings.
    Reads from environment variables and .env file.
    """
    
    # Exchange Configuration
    EXCHANGE: str
    API_KEY: SecretStr
    API_SECRET: SecretStr
    
    # Risk Management
    RISK_PER_TRADE: float = 0.01
    MAX_OPEN_POSITIONS: int = 5
    DAILY_LOSS_LIMIT: float = 0.03
    MAX_PORTFOLIO_HEAT: float = 0.05
    
    # Strategy Parameters
    ATR_PERIOD: int = 14
    DONCHIAN_PERIOD: int = 20
    ATR_STOP_MULTIPLIER: float = 1.5
    TP1_ATR_MULTIPLIER: float = 3.0
    TP2_ATR_MULTIPLIER: float = 6.0
    VOLUME_MA_PERIOD: int = 20
    VOLUME_THRESHOLD: float = 1.5
    
    # Infrastructure
    REDIS_URL: str = "redis://localhost:6379"
    DATABASE_URL: str
    LOG_LEVEL: str = "INFO"
    
    # Safety
    DRY_RUN: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

settings = Settings()
