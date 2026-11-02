# Strategic Alignment and the Foresight Gap in National AI Policies

Code and data for the paper

> Azin, M. H. and Zandhessami, H. *Strategic Alignment and the Foresight Gap in National AI Policies*. Manuscript under review.

The pipeline codes 25 national AI strategies (2017–2026) paragraph by paragraph with open-weight language models running locally, validates the coding in several independent ways, and measures how strongly each strategy links its **objectives**, its **foresight activities** and its **policy instruments**.

Everything runs on an ordinary computer without a graphics card and uses only free, open-source software and open-weight models. Every number in the paper can be reproduced from this repository.

---

## Main findings

- Strategies link objectives to instruments far more often than they link either to foresight (mean objective–instrument density 0.58, against 0.11–0.38 for links involving foresight).
- Explicit foresight activities are rare. They appear in 0.2–1.1 percent of all paragraphs, and 8–15 of the 25 strategies contain none.
- Second-generation strategies do not integrate foresight better. The European Union is the main exception.
- Strategy typologies built from link counts largely reflect document length and disappear after rarefaction.

---

## Repository structure

```
.
├── sources.csv                 list of the 25 documents with titles and years
├── 03_codebook.yaml            codebook (categories, rules, strength scale)
├── 01_download.py ... 14_tables.py   pipeline scripts (see below)
├── data/                       all intermediate and final results
├── figures/
│   ├── labeled/                figures with all text
│   └── clean/                  the same figures without any text
├── requirements.txt            exact Python package versions
├── CITATION.cff
└── LICENSE
```

The strategy PDFs are **not** included, because they belong to the governments that published them. The file `data/01_inventory.csv` lists the SHA-256 fingerprint of every PDF we used, so you can confirm that you have exactly the same files. For the same reason, the full paragraph texts (`data/02_paragraphs.csv`) are not included. Script 02 recreates them identically from the PDFs.

---

## Pipeline

Run the scripts in this order. Each script reads the outputs of the previous ones from `data/`.

| Step | Script | What it does | Main output |
|---|---|---|---|
| 1 | `01_download.py` | Checks every PDF, counts pages and text, stores SHA-256 fingerprints | `data/01_inventory.csv` |
| 2 | `02_extract_text.py` | Extracts text, removes headers and footers, splits into 6,184 paragraphs | `data/02_paragraphs.csv` |
| 3 | `03_codebook.yaml` | Codebook with 9 objectives, 6 foresight types, 7 instruments and a 0–3 strength scale | |
| 4 | `04_coder_A_ollama.py` | Coder A (Qwen2.5 7B) codes all paragraphs with evidence quotes | `data/04_coder_A.jsonl` |
| 4b | `04b_retry_long.py` | Recodes long paragraphs that exceeded the output limit | `data/04_coder_A.jsonl` |
| 5 | `05_coder_B_embeddings.py` | Coder B (multilingual sentence embeddings, no language model) | `data/05_coder_B.jsonl` |
| 6a | `06a_coder_C.py` | Coder C (Llama 3.1 8B) codes a random sample of 500 paragraphs | `data/06_coder_C.jsonl` |
| 6a retry | `06b_retry_C.py` | Recodes long paragraphs for coder C | `data/06_coder_C.jsonl` |
| 6b | `06b_reliability.py` | Agreement between coders (kappa, AC1) and checks of expectations set in advance | `data/06_reliability.csv`, `data/06_known_facts.csv` |
| 4c | `04c_foresight_detector.py` | Two-stage foresight detector (multilingual keywords, then a narrow question to both models) | `data/04c_foresight_A.jsonl`, `data/04c_foresight_C.jsonl` |
| 7 | `07_matrices.py` | Element counts and link cells for each document | `data/07_elements.csv`, `data/07_cells.csv` |
| 7b | `07b_foresight.py` | Link densities and foresight mediation under the strict and broad foresight definitions | `data/07b_foresight_docs.csv` |
| 12 | `12_gen1_vs_gen2.py` | Comparison of the 8 countries observed in both generations | `data/12_doc_measures.csv`, `data/12_summary.csv` |
| 11 | `11_clustering.py` | Ward clustering with silhouette and bootstrap stability | `data/11_clusters.csv` |
| 10b | `10b_rarefaction.py` | Rarefaction to 47 and 120 paragraphs per document | `data/10b_rarefied.csv` |
| 13 | `13_figures.py`, `13b_figure1.py` | All figures, each in a labeled and a clean version | `figures/` |
| 14 | `14_tables.py` | All tables of the paper | `data/14_*.csv` |

Scripts `08_closures.py`, `09_network_metrics.py` and `10_sensitivity.py` belong to a first round of analysis that used the foresight codes of the general coders. That round showed that these codes were unreliable, which led to the dedicated foresight detector (`04c`). The scripts are kept for transparency but are not used for the results in the paper.

---

## How to reproduce

### 1. Requirements

- Linux or Windows with WSL (tested on Ubuntu 24.04 under WSL 2)
- Python 3.12
- [Ollama](https://ollama.com) for running the language models locally
- About 25 GB of free disk space and 16 GB of RAM or more

### 2. Setup

```bash
git clone https://github.com/mhazin90/ai-strategy-foresight-gap.git
cd ai-strategy-foresight-gap
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
```

The model versions used in the paper are `qwen2.5:7b` (Ollama digest `845dbda0ea48`) and `llama3.1:8b` (Ollama digest `46e0c10c039e`). The embedding model for coder B is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

### 3. Documents

Download the 25 strategy PDFs listed in `sources.csv` from the official government websites or the OECD AI Policy Observatory, save each one as `pdfs/<id>.pdf` (for example `pdfs/DEU_2018.pdf`), and check the fingerprints against `data/01_inventory.csv`.

### 4. Run

```bash
python 01_download.py
python 02_extract_text.py
python 04_coder_A_ollama.py
python 04b_retry_long.py
python 05_coder_B_embeddings.py
python 06a_coder_C.py
python 06b_retry_C.py
python 06b_reliability.py
python 04c_foresight_detector.py --candidates-only
python 04c_foresight_detector.py --model qwen2.5:7b --tag A
python 04c_foresight_detector.py --model llama3.1:8b --tag C
python 07_matrices.py
python 07b_foresight.py
python 12_gen1_vs_gen2.py
python 11_clustering.py
python 10b_rarefaction.py
python 13_figures.py
python 13b_figure1.py
python 14_tables.py
```

All model calls use temperature 0 and a fixed seed, so repeated runs give the same codes. All coding scripts save after every paragraph and continue where they stopped if interrupted.

### 5. Running time

On a 20-thread CPU without a graphics card, coder A needs about 30 hours for all 6,184 paragraphs, coder C about 4 hours for its sample, and the foresight detector about 2 hours for both models. All other steps take seconds to minutes.

---

## Data

All files in `data/` are produced by the scripts. The most important ones are

- `04_coder_A.jsonl`, `05_coder_B.jsonl`, `06_coder_C.jsonl` with the codes for every paragraph, including evidence quotes and raw model answers
- `04c_foresight_A.jsonl`, `04c_foresight_C.jsonl` with the decisions of the foresight detector
- `12_doc_measures.csv` with all measures for every document
- `14_table*.csv` with the tables of the paper

---

## License

The code is released under the MIT License (see `LICENSE`). The data produced by the pipeline are released under the Creative Commons Attribution 4.0 International License (CC BY 4.0). The strategy documents themselves remain the property of their publishers.

## Citation

If you use this code or data, please cite the paper and this repository (see `CITATION.cff`).

## Contact

Hessam Zandhessami, h.zandhessami@iau.ir
