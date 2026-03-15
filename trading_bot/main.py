from trading_bot.pipeline import TradingPipeline

if not TradingPipeline.PAPER_ONLY or not TradingPipeline.DRY_RUN:
    raise RuntimeError("LIVE TRADING NOT PERMITTED — PAPER_ONLY and DRY_RUN must both be True")

def main() -> None:
    print("Trading Bot initialized safely.")

if __name__ == "__main__":
    main()
