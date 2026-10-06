install:
	pip install -r requirements.txt

# The point-in-time universe (membership spells) is downloaded and cached by
# src/data/universe.py the first time a panel is built; there is no separate
# build step. The old `universe` target pointed at a script that no longer exists.

test:
	python3 -m pytest -q project/tests

run:
	cd project && python3 -m scripts.run_backtest --config configs/default.yaml

# Reproduces docs/BIAS.md end to end.
bias: bias-dating bias-survivorship

bias-dating:
	cd project && python scripts/pit_vs_naive.py

bias-survivorship:
	cd project && python scripts/survivorship.py

# Gates one to three of the validation protocol.
gates:
	cd project && python3 scripts/gate_ic_series.py && python3 scripts/gates.py

.PHONY: install test run bias bias-dating bias-survivorship gates
