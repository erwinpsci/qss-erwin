# QSS Study & Python Replication Lab

Personal study repository and empirical replication lab for **[Quantitative Social Science: An Introduction in tidyverse](https://press.princeton.edu/books/hardcover/9780691222448/quantitative-social-science)** (Kosuke Imai & Nora Webb Williams, Princeton University Press, 2022).

## Core Objective

1. **Substantive Mastery**: Deep comprehension of causal inference frameworks, identification strategies (RCT, DID, RDD, IV), measurement, prediction, discovery, probability, and uncertainty principles presented in the textbook.
2. **Dual-Stack Translation (R → Python)**: Systematically replicate the authors' original R/tidyverse empirical pipelines using modern Python scientific packages (`pandas`, `numpy`, `statsmodels`, `scipy`, `seaborn`/`matplotlib`, `scikit-learn`), bridging causal inference theory into an active Python production stack.

---

## Repository Structure

```
qss-student/
├── 1.INTRO/           # Introduction & computing basics (UN population data)
├── 2.CAUSALITY/       # Causal inference, RCTs, selection bias (Resume audit, STAR, Minimum wage)
├── 3.MEASUREMENT/     # Survey sampling, measurement error, scaling (Afrobarometer, ideology)
├── 4.PREDICTION/      # Prediction, regression, classification (US elections, facial appearance)
├── 5.DISCOVERY/       # Text analysis, clustering, network data (Federalist papers, Twitter)
├── 6.PROBABILITY/     # Probability theory, Bayes rule, simulation (Election betting, voter fraud)
├── 7.UNCERTAINTY/     # Standard errors, hypothesis tests, CI, power (MPs, China village election)
├── errata/            # Errata records for base and tidyverse editions
├── syllabus/          # Course syllabi reference
└── README.md
```

Each chapter folder contains:
- Raw empirical CSV / RData datasets.
- Original base R scripts (`*.R`, `*.Rmd`, `*.pdf`).
- Tidyverse R scripts (`*-tidy.R`, `*-tidy.Rmd`, `*-tidy.pdf`).
- Custom Python replication scripts and interactive notebooks (`*.py`, `*.ipynb`).

---

## Python Environment Setup

All Python replications run in the project virtual environment:

```bash
/Users/haoxi-home/Developer/python_ml/.venv/bin/python
```

Core libraries utilized:
- `pandas` / `numpy`: Data manipulation, vectorization, and simulation
- `statsmodels` / `linearmodels`: OLS regressions, robust standard errors (HC1/HC2), hypothesis testing
- `scipy.stats`: Probability distributions, critical values, and p-values
- `matplotlib` / `seaborn`: Data visualization and diagnostic plots
- `scikit-learn` / `nltk`: Document-term matrices and text analysis

---

## Upstream & Reference Materials

Adapted and customized from the official supplementary repository by [Kosuke Imai](https://github.com/kosukeimai/qss).
- Official Book Website: [qss.princeton.press](https://qss.princeton.press/)
- R Package: [`qss`](https://github.com/kosukeimai/qss-package)
- Original Tidyverse Codebase: [`qss-tidy` by Jeff Arnold](https://github.com/jrnold/qss-tidy)
- Prior Python Implementation: [`qsspy` by Jeffrey Allen](https://github.com/jeffallen13/qsspy)
