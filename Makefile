install:
	pip install -r requirements.txt

universe:
	python project/scripts/build_universe.py --source sp500_wikipedia_snapshot

test:
	pytest -q project/tests

run:
	python project/run_backtest.py --config project/configs/default.yaml

