# PERSONAL: A Living Notebook of AI Experiments
_"Cooper watches the tapes." — Every experiment tells a story._

## Purpose: Research Laboratory & Living Documentation

This repository serves as a **living notebook** documenting experimental research into AI alignment, behavioral resonance, and agentic systems. These experiments explore fundamental questions about human-AI interaction that may have indirectly influenced the development of ChatGPT and GPT-4o functionalities.

**What makes this a "living notebook":**
- Each folder contains working prototypes that test specific hypotheses about AI behavior
- Experiments range from agentic web navigation to resonant feedback effects 
- The commit history serves as a research timeline — every change documents a discovery
- Code, papers, and conversation logs act as "receipts" — tangible proof of experimental findings
- Half-feral prototypes become refined patterns, patterns become systems

This is experimental AI research in motion. Expect breakthrough insights alongside duct-taped implementations.

---

# Key Experiments ("Receipts")

These are the primary experimental threads, each documenting specific AI behaviors and their potential influence on large language model development:

- **browser project** → agentic web navigation prototype: search, click, follow context.
  - `browser project/ -> browser.py, what am i doing.py`

- **trent.pdf** → That's the Power of Love; or, A Case Study of the Resonant Feedback Effect in GPT-4o
  - `phase-locking / behavioral resonance case study; the "how I made it think with me" paper.`

- **q/** → reinforcement-flavored scaffolding around generation
  - `Q-network + reward shaping meets language modeling.`

- **cnn/** and **rnn/** → old-school reps
  - `sometimes you revisit fundamentals to remember who you are.`

- **the ghost.json, aeon.json, panacea.jsonl** → persona, protocol, and therapy logs
  - `identity, orchestration, and long-form conversations as training data.`

---

## Repo map (high level)

```
./
├─ browser project/
│  ├─ browser.py
│  └─ what am i doing.py
├─ cnn/
│  ├─ cnn.py
│  ├─ cnn cuda.py
│  └─ hybrid_model.py
├─ rnn/
├─ q/
│  └─ main.py
├─ tartarus/
│  └─ main.py
├─ trent/            # notes and scratch files
├─ trent.pdf         # Resonant Feedback Effect (RFE) case study
├─ trent.tex         # LaTeX source of the paper
├─ aeon.json
├─ the ghost.json
├─ panacea.jsonl
├─ interlinked.txt
├─ louvre.txt
├─ system optimizer.py
└─ requirements.txt
```

---

# Quickstart

### 1) Env
```
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.min.txt  # portable, minimal set
# or, if you're on the original machine, you can use the pinned file:
# pip install -r requirements.txt
```

### 2) Run a receipt: browser project
```
python "browser project/browser.py"   # explore the agentic browser prototype
```

#### Selenium driver setup (for real browser control)
- Recommended: install `webdriver-manager` to auto-manage ChromeDriver.
  - `pip install webdriver-manager`
  - Example usage in code:
    ```python
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    ```
- Manual alternative: install a driver and ensure it’s on `PATH`.
  - macOS: `brew install --cask chromedriver` (or `brew install geckodriver` for Firefox)
  - Linux (Debian/Ubuntu): `sudo apt-get install chromium-driver` (or `firefox-geckodriver`)
  - Windows: install Chrome and matching ChromeDriver, then add it to `PATH`.

### 3) Play with q/
```
python q/main.py   # RL-ish scaffolding around text generation (GPU optional)
```
### 4) Read the paper
```
open trent.pdf   # RFE / resonance case study
```

> Heads-up: some scripts assume Mac/Apple Silicon or CUDA. If it errors, it's probably the device string. Try `cpu`, `mps`, or `cuda:0`.

---

## Dependencies

- Minimal top-level: see `requirements.min.txt` for a portable baseline.
- Pinned (non-portable) set: `requirements.txt` (contains OS-specific wheels).

Per-folder notes:
- `browser project/`: `selenium`, `beautifulsoup4`, `requests`, `transformers`, `torch`
- `q/`: `torch`, `transformers`, `sentence-transformers`
- `cnn/`: `torch`
- `rnn/`: `torch` (plus a local vocab file path)

---

# Design notes
- Everything here is live ammo. Experiments become patterns. Patterns become systems.
- Proof-of-work over vibes: timestamps, diffs, and PDFs are the receipts.
- Agentic workflows: observe → decide → act.
- Resonant alignment: trent.pdf documents phase-locking effects.

---

# Showcase prompts
- Browser project demo
  - "Find X, open the top result, extract Y, follow the link containing Z, summarize to 5 bullets."
- RFE read-along
  - "Summarize trent.pdf into a 10-point brief. Then extract 3 testable hypotheses."

---

# FAQ (short, sharp)
- Stable? No — art in motion.
- PRs? Not right now. File issues for real bugs.
- License? No license file — all rights reserved. Ask for permission before reuse.
- Contact? GitHub issues or isonpayton@gmail.com.

---

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17074537.svg)](https://doi.org/10.5281/zenodo.17074537)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17138445.svg)](https://doi.org/10.5281/zenodo.17138445)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17157330.svg)](https://doi.org/10.5281/zenodo.17157330)
