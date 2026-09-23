# QSS Study Notebook Reference

- **Title:** QSS
- **Notebook URL:** https://notebooklm.google.com/notebook/4466791f-3638-4c1d-b52a-c7f3f742896e
- **Notebook UUID:** `4466791f-3638-4c1d-b52a-c7f3f742896e`
- **CLI Aliases:** `qss`, `qss-tidyverse`

## Ingested Sources (9 Chapter PDFs)

| Chapter | Title | Source UUID |
|---|---|---|
| 0 | 0.Table of Contents.pdf | `06cd06dc-2e50-4099-9022-e681895d44f4` |
| 1 | 1.Introduction.pdf | `65c635d3-da2f-4a88-915d-aa054c085ecf` |
| 2 | 2.Causality.pdf | `279efdba-cc65-4b26-975f-f20935b4e5e2` |
| 3 | 3.Measurement.pdf | `a8e0de09-1c8f-456e-b955-cfa9c7925719` |
| 4 | 4.Prediction.pdf | `ce3e050b-a0bf-41a8-a841-969c990c1e3d` |
| 5 | 5.Discovery.pdf | `43b25e31-263d-45a7-8a40-7b4dfc1dc8d1` |
| 6 | 6.Probability.pdf | `08404162-e77c-4ade-937b-5fb827ef816c` |
| 7 | 7.Uncertainty.pdf | `5063e5d7-52eb-4e7a-9d0a-7abd7905b0b9` |
| 8 | 8.Next and Index.pdf | `ba19cfbb-3d11-47e6-b573-fab02b6dda42` |

## Usage Commands

### CLI (`nlm`)
```bash
# Query the textbook
nlm notebook query qss "How is average treatment effect defined in Chapter 2?"

# Query with fresh conversation context
nlm notebook query qss "Explain diff-in-diff" --new-conversation

# List sources
nlm source list qss

# Get notebook summary and suggested study topics
nlm notebook describe qss
```

### MCP (`gemini-notebook-mcp`)
```python
notebook_query(notebook_id="4466791f-3638-4c1d-b52a-c7f3f742896e", query="<question>")
```
