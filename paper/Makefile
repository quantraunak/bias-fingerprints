# Build the paper. Requires TeX Live and matplotlib.
PAPER = bias_fingerprints

all: $(PAPER).pdf

figures/fingerprints.pdf: ../project/reports/pit_vs_naive.csv ../project/reports/survivorship.csv
	python3 make_figures.py

$(PAPER).pdf: $(PAPER).tex figures/fingerprints.pdf
	latexmk -pdf -interaction=nonstopmode $(PAPER).tex
	latexmk -c

arxiv: $(PAPER).pdf
	rm -rf arxiv && mkdir -p arxiv/figures
	cp $(PAPER).tex arxiv/
	cp figures/fingerprints.pdf arxiv/figures/
	cd arxiv && tar czf ../$(PAPER)-arxiv.tar.gz .
	@echo "arXiv submission tarball: $(PAPER)-arxiv.tar.gz"

clean:
	latexmk -C
	rm -rf arxiv $(PAPER)-arxiv.tar.gz

.PHONY: all arxiv clean
