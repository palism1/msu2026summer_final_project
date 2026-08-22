# IEEE conference version of the project report

`PROJECT_REPORT.tex` is generated from `../PROJECT_REPORT.md` by `md2tex.py`
and compiled with the IEEEtran class in conference mode (IEEE template, June 2024).

Regenerate and build:

    python3 docs/ieee/md2tex.py docs/PROJECT_REPORT.md docs/ieee/PROJECT_REPORT.tex
    cd docs/ieee && tectonic PROJECT_REPORT.tex

Any LaTeX engine with IEEEtran works. Overleaf compiles the `.tex` and `.cls`
as they are. The abstract and the keywords live in `md2tex.py`, not in the
markdown. Edit the markdown for everything else, then regenerate.
