"""Convert docs/PROJECT_REPORT.md into an IEEEtran conference paper."""
import re, sys, pathlib

src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
lines = src.read_text().split("\n")
if lines[0] == "---":
    lines = lines[lines.index("---", 1) + 1:]

UNI = {"±": r"$\pm$", "×": r"$\times$", "→": r"$\rightarrow$", "−": r"$-$",
       "–": "--", "—": "---", "’": "'", "‘": "`", "“": "``", "”": "''",
       "á": r"\'a", "é": r"\'e", "í": r"\'{\i}", "ó": r"\'o", "ú": r"\'u",
       "ñ": r"\~n", "ä": r'\"a', "ö": r'\"o', "ü": r'\"u', "β": r"$\beta$"}

def esc(t):
    t = t.replace("\\", r"\textbackslash{}")
    for k, v in UNI.items():
        t = t.replace(k, v)
    t = re.sub(r"([&%#_])", r"\\\1", t)
    t = re.sub(r'"([^"]*)"', r"``\1''", t)
    t = t.replace("$\\pm$", "$\\pm$")  # already math
    t = re.sub(r"(?<!\\)\$(?!\\)", r"\\$", t) if False else t
    return t

def inline(t):
    """bold, italic, code, citations; escape the rest."""
    out, pos = [], 0
    for m in re.finditer(r"(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*|\[(\d+(?:,\s*\d+)*)\])", t):
        out.append(esc(t[pos:m.start()]))
        tok = m.group(1)
        if tok.startswith("**"):
            out.append(r"\textbf{" + inline(tok[2:-2]) + "}")
        elif tok.startswith("`"):
            out.append(r"\texttt{" + esc(tok[1:-1]).replace(" ", "~") + "}")
        elif tok.startswith("*"):
            out.append(r"\textit{" + inline(tok[1:-1]) + "}")
        else:
            nums = [n.strip() for n in m.group(2).split(",")]
            if any(int(n) < 1 or int(n) > 40 for n in nums):
                out.append(esc(tok))
            else:
                out.append(r"\cite{" + ",".join("ref" + n for n in nums) + "}")
        pos = m.end()
    out.append(esc(t[pos:]))
    return "".join(out)

# ---------- pass 1: split into blocks ----------
out = []
buf = []
refs = {}
mode = None  # None | "refs"
i = 0
in_list = None

def flush():
    global buf
    if buf:
        out.append(inline(" ".join(buf)) + "\n")
        buf = []

def close_list():
    global in_list
    if in_list:
        out.append("\\end{%s}" % in_list)
        in_list = None

abstract_done = False
while i < len(lines):
    ln = lines[i]
    # references section
    if ln.startswith("## 9. References"):
        flush(); close_list(); mode = "refs"; i += 1; continue
    if mode == "refs":
        if ln.startswith("## "):
            mode = None
        else:
            m = re.match(r"^\[(\d+)\] (.*)", ln)
            if m:
                refs[m.group(1)] = [m.group(2)]
            elif ln.strip() and refs:
                refs[max(refs, key=int)].append(ln.strip())
            i += 1; continue
    if ln.startswith("|"):
        flush(); close_list()
        rows = []
        while i < len(lines) and lines[i].startswith("|"):
            cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            if not all(re.fullmatch(r"-+", c) for c in cells):
                rows.append(cells)
            i += 1
        out.append(("TABLE", rows))
        continue
    m = re.match(r"^(#+) (.*)", ln)
    if m:
        flush(); close_list()
        level, title = len(m.group(1)), m.group(2)
        nm = re.match(r"^(\d+(?:\.\d+)?)\.? ", title)
        secnum = nm.group(1) if nm else None
        title = re.sub(r"^\d+(\.\d+)?\.? ", "", title)
        if level == 1:
            out.append(("TITLE", title))
        elif level == 2:
            if title.startswith("Appendices"):
                out.append(("SECTION", ("Reproducibility", None)))
            else:
                out.append(("SECTION", (title, secnum)))
        else:
            if title.startswith("Appendix "):
                t = re.sub(r"^Appendix [A-C]: ", "", title)
                if t != "Hyperparameters":
                    out.append(("SUBSECTION", (t, None)))
            else:
                out.append(("SUBSECTION", (title, secnum)))
        i += 1; continue
    if ln.strip() == "---":
        flush(); close_list(); i += 1; continue
    lm = re.match(r"^(\s*)(\d+\.|-) (.*)", ln)
    if lm and not ln.startswith("   "):
        flush()
        kind = "enumerate" if lm.group(2) != "-" else "itemize"
        if in_list != kind:
            close_list(); out.append("\\begin{%s}" % kind); in_list = kind
        text = lm.group(3)
        i += 1
        while i < len(lines) and lines[i].startswith("   ") and lines[i].strip():
            text += " " + lines[i].strip(); i += 1
        out.append("\\item " + inline(text))
        continue
    if ln.strip() == "":
        flush(); close_list(); i += 1; continue
    buf.append(ln.strip()); i += 1
flush(); close_list()

# ---------- pass 2: emit ----------
def table(rows, caption, label):
    ncol = len(rows[0])
    wide = ncol >= 4 or max(len(" ".join(r)) for r in rows) > 55
    SHORT = {"MedSAM + LoRA (min-max norm)": "MedSAM + LoRA (min-max)",
             "MedSAM + LoRA (ImageNet norm, as published)": "MedSAM + LoRA (as published)",
             "MedSAM + LoRA (ImageNet norm, jitter control)": "MedSAM + LoRA (jitter control)"}
    SHORT.update({"r = 4, alpha = 8, dropout 0.1, merged qkv projection of each encoder attention block":
                  "r = 4, alpha = 8, dropout 0.1, merged qkv projection",
                  "ImageNet stats (SAM, U-Net); per-image min-max [0,1] (corrected MedSAM arms)":
                  "ImageNet stats (SAM, U-Net); min-max [0,1] (MedSAM arms)",
                  "NVIDIA A100-SXM4-40GB (Google Colab)": "NVIDIA A100-SXM4-40GB (Colab)"})
    rows = [[SHORT.get(c, c) for c in r] for r in rows]
    if ncol == 2:
        wide = False
    env = "table*" if wide else "table"
    spec = "lp{0.62\\columnwidth}" if ncol == 2 else "l" + "c" * (ncol - 1)
    body = []
    for r, row in enumerate(rows):
        cells = [inline(c) for c in row]
        if r == 0:
            cells = [r"\textbf{" + c + "}" for c in cells]
        body.append(" & ".join(cells) + r" \\")
        if r == 0:
            body.append(r"\hline")
    size = r"\footnotesize"
    return "\n".join([
        r"\begin{%s}[t]" % env, r"\centering", size,
        r"\caption{%s}" % caption, r"\label{%s}" % label,
        r"\begin{tabular}{%s}" % spec, r"\hline"] + body +
        [r"\hline", r"\end{tabular}", r"\end{%s}" % env, ""])

CAPTIONS = [
    "PraNet five-split protocol: datasets, roles, and image counts",
    "Models compared: backbone, trainable parameters, and role",
    "The MedSAM three-arm experiment",
    "Accuracy and cost of the trained, prompt-free models (mean $\\pm$ std over seeds 42, 43, 44; one A100)",
    "Per-split mean Dice over three seeds",
    "Matched pairings: one factor changed per comparison (unseen mDice)",
    "Oracle-box upper bound: untrained models prompted with a ground-truth box (separate tier)",
    "Hyperparameters",
]
tex = []
tcount = 0
title = None
for item in out:
    if isinstance(item, tuple):
        kind, val = item
        if kind == "TITLE":
            title = inline(val)
        elif kind == "SECTION":
            t, n = val
            tex.append("\n\\section{%s}%s" % (inline(t), "\\label{sec:%s}" % n if n else ""))
        elif kind == "SUBSECTION":
            t, n = val
            tex.append("\n\\subsection{%s}%s" % (inline(t), "\\label{sec:%s}" % n if n else ""))
        elif kind == "APPSECTION":
            tex.append("\n\\section{%s}" % inline(val))
        elif kind == "TABLE":
            cap = CAPTIONS[tcount] if tcount < len(CAPTIONS) else "Table"
            tex.append(table(val, cap, "tab:%d" % (tcount + 1)))
            tcount += 1
    else:
        tex.append(item)

body = "\n".join(tex)
body = re.sub(r"Section (\d+(?:\.\d+)?)", r"Section~\\ref{sec:\1}", body)

# drop the author/venue lines that the markdown title block carried
body = re.sub(r"\\textbf\{Project Report, Research Method in Computing\}\n+", "", body)
body = re.sub(r"\\textbf\{MSU Summer 2026, Master's Final Project\}\n+", "", body)
body = re.sub(r"^Mikko Palis\n+", "", body, flags=re.M)

# bibliography
bib = ["\\begin{thebibliography}{00}"]
for n in sorted(refs, key=int):
    entry = " ".join(refs[n])
    entry = re.sub(r"\s*\((Kvasir-SEG|CVC-ClinicDB|CVC-ColonDB|ETIS-LaribPolypDB|CVC-300 / EndoScene)\)\s*$", "", entry)
    entry = inline(entry)
    entry = re.sub(r"https?://\S+", lambda m: r"\url{" + m.group(0).replace(r"\_", "_") + "}", entry)
    bib.append("\\bibitem{ref%s} %s" % (n, entry))
bib.append("\\end{thebibliography}")

ABSTRACT = r"""Specialist polyp segmentation networks fit their training distribution tightly and lose accuracy on data from other clinics. This study asks whether the Segment Anything Model (SAM), adapted with Low-Rank Adaptation (LoRA) while its backbone stays frozen, matches a specialist U-Net on the PraNet benchmark and loses less accuracy on the three datasets neither model trained on. Six model variants were trained under one protocol, three seeds each, on one NVIDIA A100, and four untrained oracle-box baselines were evaluated, for 22 runs in total. SAM-ViT-H + LoRA scores highest on the unseen splits (0.806 mean Dice against U-Net's 0.755) with a seen-to-unseen drop about half of U-Net's (0.082 against 0.145), while it trains 830,177 parameters, about 3\% of the 24.4M that U-Net trains end-to-end. The specialist keeps the edge on seen data (0.900) and trains twelve times faster. A controlled three-arm experiment splits MedSAM's 0.100 unseen deficit against SAM-ViT-B into a 0.069 preprocessing artifact in the author's own pipeline and a 0.036 residual for the checkpoint at matched backbone size. With a ground-truth box and its native min-max inputs, untrained MedSAM reaches 0.925 unseen Dice, the highest number in the oracle tier, and the size of that lift (+0.200 against +0.099 for generic SAM) quantifies its prompt dependence. All numbers rest on one benchmark family and three seeds, with no significance test."""

KEYWORDS = "polyp segmentation, Segment Anything Model, LoRA, parameter-efficient fine-tuning, MedSAM, cross-dataset generalization, colonoscopy"

doc = r"""\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{url}
\usepackage{booktabs}
\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}
\begin{document}

\title{%s}

\author{\IEEEauthorblockN{Mikko Palis}
\IEEEauthorblockA{\textit{Department of Computer Science} \\
\textit{Montclair State University}\\
Montclair, NJ, USA \\
mikkopalis@gmail.com}
}

\maketitle

\begin{abstract}
%s
\end{abstract}

\begin{IEEEkeywords}
%s
\end{IEEEkeywords}
%s

%s

\end{document}
""" % (title, ABSTRACT, KEYWORDS, body, "\n".join(bib))
dst.write_text(doc)
print("wrote", dst, "tables:", tcount, "refs:", len(refs))
