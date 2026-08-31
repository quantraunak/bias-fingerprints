install:
	pip install -r requirements.txt

universe:
	python project/scripts/build_universe.py --source sp500_wikipedia_snapshot

test:
	pytest -q project/tests

run:
	python project/run_backtest.py --config project/configs/default.yaml

# Reproduces docs/BIAS.md end to end.
bias: bias-dating bias-survivorship

bias-dating:
	cd project && python scripts/pit_vs_naive.py

bias-survivorship:
	cd project && python scripts/survivorship.py

# Writes the redistributable point-in-time layer to project/dist/.
dataset:
	cd project && python scripts/export_dataset.py

.PHONY: install universe test run bias bias-dating bias-survivorship dataset
