# ICECCME 2026 talk
#   make          -> slides.pdf (projector deck)
#   make notes    -> slides-notes.pdf (speaker notes, one page per slide)
#   make all      -> both
#   make clean    -> remove build artifacts

LATEXMK = latexmk -pdf -interaction=nonstopmode -halt-on-error

.PHONY: all slides notes clean

all: slides notes

slides: slides.pdf

notes: slides-notes.pdf

slides.pdf: slides.tex preamble.tex ../figures/specdec_configs_vs_baseline_metrics.pdf
	$(LATEXMK) slides.tex

slides-notes.pdf: slides-notes.tex slides.tex preamble.tex
	$(LATEXMK) slides-notes.tex

clean:
	latexmk -C slides.tex slides-notes.tex
	rm -f *.nav *.snm *.vrb
