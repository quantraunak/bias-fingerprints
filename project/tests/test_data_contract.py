import pytest
from pathlib import Path

from src.data.universe import load_universe


def test_invalid_universe_source_raises():
    with pytest.raises(ValueError, match="universe.source"):
        load_universe("invalid", Path("data/raw/universe.csv"))


def test_file_snapshot_requires_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_universe("file_snapshot", tmp_path / "missing.csv")


def test_file_snapshot_reads_tickers(tmp_path):
    p = tmp_path / "universe.csv"
    tickers = [f"T{i:02d}" for i in range(25)]
    p.write_text("ticker\n" + "\n".join(tickers) + "\n")
    result = load_universe("file_snapshot", p, benchmark="SPY")
    assert "T00" in result.tickers
    assert "SPY" in result.tickers
    assert result.source == "file_snapshot"
