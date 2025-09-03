all: paper.pdf

paper.pdf: paper.md paper.bib
	pandoc \
		--from markdown \
		--output paper.pdf \
		--citeproc \
		--bibliography=paper.bib \
		--template=/usr/share/pandoc/data/templates/default.latex \
		--pdf-engine=xelatex \
		paper.md
