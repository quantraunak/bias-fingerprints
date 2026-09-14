install:
	pip install -r requirements.txt

universe:
	python project/scripts/build_universe.py --source sp500_wikipedia_snapshot

test:
	python3 -m pytest -q project/tests

run:
	python project/run_backtest.py --config project/configs/default.yaml

# Reproduces docs/BIAS.md end to end.
bias: bias-dating bias-survivorship

bias-dating:
	cd project && python scripts/pit_vs_naive.py

bias-survivorship:
	cd project && python scripts/survivorship.py

# Gates one to three of the validation protocol.
gates:
	cd project && python3 scripts/gate_ic_series.py && python3 scripts/gates.py

.PHONY: install universe test run bias bias-dating bias-survivorship gates
