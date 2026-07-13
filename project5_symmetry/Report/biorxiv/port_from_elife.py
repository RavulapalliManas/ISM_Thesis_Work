import re, pathlib
src = pathlib.Path("/Volumes/Crucial X6/Thesis_work/project5_symmetry/Report/elife/main_best.tex").read_text()

# --- extract title ---
title = re.search(r'\\title\{(.+?)\}', src, re.S).group(1).strip()
title = re.sub(r'\s+', ' ', title)

# --- extract abstract inner text ---
ab,  = re.findall(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', src, re.S)
ab = ab.replace(r'\noindent', '').strip()

# --- extract body: from \section{Introduction} up to \bibliography ---
body = src[src.index(r'\section{Introduction}'):src.index(r'\bibliography')].rstrip()

# --- transforms for two-column HenriquesLab class ---
body = body.replace(r'\section{',   r'\section*{')
body = body.replace(r'\subsection{',r'\subsection*{')
# wide floats span both columns
body = re.sub(r'\\begin\{figure\}\[[^\]]*\]', r'\\begin{figure*}', body)
body = body.replace(r'\end{figure}', r'\end{figure*}')
body = re.sub(r'\\begin\{table\}\[[^\]]*\]', r'\\begin{table*}', body)
body = body.replace(r'\end{table}', r'\end{table*}')

preamble = r"""%% catoptions (2014) and xwatermark (2012), both pulled in by the class, patch \begin{document}
%% in a way that conflicts with LaTeX's 2020 hook system ("Extra \endgroup" on modern TeX Live)
%% -- pin the format to pre-hook behaviour.
\RequirePackage[2020-01-01]{latexrelease}
\documentclass[times, twoside]{zHenriquesLab-StyleBioRxiv}
\usepackage{booktabs}
\usepackage{amsmath}

\graphicspath{{figures/}}
\leadauthor{Ravulapalli}

\begin{document}

\title{%s}
\shorttitle{Cognitive maps are symmetry quotients}

\author[1,\Letter]{Manas Venkata Sai Ravulapalli}
\affil[1]{Ashoka University, Sonipat, Haryana, India}

\maketitle

\begin{abstract}
%s
\end{abstract}

\begin{keywords}
predictive learning | cognitive map | hippocampus | symmetry | place cells | head direction
\end{keywords}

\begin{corrauthor}
manasvenkatasai.ravulapalli\_ug2023\at ashoka.edu.in
\end{corrauthor}

""" % (title, ab)

tail = r"""

\bibliography{../references}

\end{document}
"""

out = preamble + body + tail
pathlib.Path("/Volumes/Crucial X6/Thesis_work/project5_symmetry/Report/biorxiv/main.tex").write_text(out)
print("wrote biorxiv/main.tex")
print("title:", title[:70], "...")
print("abstract words:", len(ab.split()))
print("figure* envs:", out.count(r'\begin{figure*}'), " table* envs:", out.count(r'\begin{table*}'))
print("section*:", out.count(r'\section*{'), " subsection*:", out.count(r'\subsection*{'))
