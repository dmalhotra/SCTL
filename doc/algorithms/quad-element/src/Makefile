# Renders each standalone TikZ source to an SVG in the parent directory: pdflatex, then Ghostscript
# converts the text to outlines (so no fonts are needed to view it), then Inkscape writes the SVG.
SVG = $(patsubst %.tex,../%.svg,$(wildcard *.tex))

all: $(SVG)

../%.svg: %.tex
	pdflatex -interaction=batchmode -halt-on-error $<
	gs -q -o $*-outlines.pdf -sDEVICE=pdfwrite -dNoOutputFonts $*.pdf
	inkscape $*-outlines.pdf --export-type=svg --export-plain-svg --export-filename=$@
	$(RM) $*.aux $*.log $*.pdf $*-outlines.pdf

clean:
	$(RM) $(SVG) *.aux *.log *.pdf

.PHONY: all clean
