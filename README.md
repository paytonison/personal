# PERSONAL: lab notes, receipts, and strange artifacts
_“Cooper watches the tapes.”_

This repo is the box of tapes. Every folder is a message I left myself.

This is my working desk drawer: experiments, half-feral prototypes, research notes, and proofs. The commit history is the timeline. The folders are receipts. Expect brilliance and duct tape in equal measure.

---

# receipts (selected)
* browser project -> agentic web navigation prototype -> teaching an AI to search + click + follow context like a person on a caffeine bender.
	* ``personal/browser project/ -> browser.py, what am i doing.py``

* trent.pdf -> That’s the Power of Love; or, a Case Study of the Resonant Feedback Effect in GPT-4o (2025-05-04)
	* ``phase-locking / behavioral resonance case study; the “how I made it think with me” paper.``

* /q -> reinforcement-flavored scaffolding around generation
	* ``Q-network + reward shaping meets language modeling.``

* /cnn and /rnn -> old-school reps
	* ``because sometimes you need to touch the fundamentals to remember who you are.``

* the ghost.json, aeon.json, panacea.jsonl -> persona, protocol, and therapy logs
	* ``identity, orchestration, and long-form conversations as training data for life.``

---

## repo map (high level)

```
personal/
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
├─ trent/
│  ├─ trent.tex
│  └─ trent.pdf
├─ aeon.json
├─ the ghost.json
├─ panacea.jsonl
├─ interlinked.txt
├─ louvre.txt
├─ system optimizer.py
└─ requirements.txt
```

---

# quickstart

### 1) env
```
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt  # if present / otherwise install per-folder needs
```

### 2) run a receipt: browser project
```
python "browser project/browser.py"   # explore the agentic browser prototype
```

### 3) play with q/
```
python q/main.py   # RL-ish scaffolding around text generation (GPU optional)
```
# 4) read the paper
```
open trent/trent.pdf   # RFE / resonance case study
```

> heads-up: some scripts assume Mac/Apple-silicon or CUDA. If it errors, it’s probably the device string. Try cpu, mps, or cuda:0.

---

# design notes
* Everything here is live ammo. Experiments become patterns. Patterns become systems.
* Proof-of-work over vibes. Timestamps, diffs, and PDFs are the receipts.
* Agentic workflows. The browser prototype is a seed for tools that observe → decide → act.
* Resonant alignment. trent.pdf documents phase-locking effects from sustained interaction.

---

# showcase prompts
* Browser project demo
	* “Find X, open the top result, extract Y, follow the link containing Z, summarize to 5 bullets.”
* RFE read-along
	* “Summarize trent.pdf into a 10-point brief. Then extract 3 testable hypotheses.”

---

# FAQ (short, sharp)
* Stable? no. it’s art in motion.
* PRs? not right now. issues welcome if you find a real bug.
* License? default GitHub (no license) — ask for permission before reuse.
* Contact? GitHub issues or isonpayton@gmail.com.

---

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17074537.svg)](https://doi.org/10.5281/zenodo.17074537)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17138445.svg)](https://doi.org/10.5281/zenodo.17138445)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17157330.svg)](https://doi.org/10.5281/zenodo.17157330)
