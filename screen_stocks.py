from __future__ import annotations

import argparse
from pathlib import Path

from predictor import screen_tickers


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen multiple tickers with the stock dashboard model.")
    parser.add_argument("--tickers", required=True, help="Comma-separated tickers")
    parser.add_argument("--period", default="5y", help="History period")
    parser.add_argument("--threshold", type=float, default=0.55, help="Signal threshold")
    parser.add_argument("--out", default="screen_results.csv", help="Output CSV path")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    df = screen_tickers(tickers, period=args.period, threshold=args.threshold)
    out_path = Path(args.out)
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
