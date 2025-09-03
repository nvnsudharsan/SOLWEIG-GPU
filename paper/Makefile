all: paper.pdf

paper.pdf: paper.md references.bib
	pandoc \
		--from markdown \
		--output paper.pdf \
		--citeproc \
		--bibliography=references.bib \
		--template=/opt/pandoc/templates/default.latex \
		--pdf-engine=xelatex \
		paper.md
