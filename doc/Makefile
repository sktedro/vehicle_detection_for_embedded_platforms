NAME=doc

ALLFILES=$(NAME).tex Makefile

all: pdf

pdf:
	pdflatex $(NAME)
	bibtex $(NAME)
	pdflatex $(NAME)
	pdflatex $(NAME)

clean:
	@ls | grep -v ".gitignore\|Makefile\|${NAME}.tex\|${NAME}.bib\|${NAME}.pdf\|src" | xargs rm -f
	rm -f src/*eps-converted-to.pdf
