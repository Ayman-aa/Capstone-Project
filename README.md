# The Cost of Intelligence: Evaluating Small Language Models on Consumer Hardware

**Authors:** Ayman Aamam, Fatima Zahra Abeddad  
**Course:** Introduction to Data Science and Artificial Intelligence  
**Date:** May 2026  
**Repository:** [Capstone-Project](https://github.com/Ayman-aa/Capstone-Project) (branch: `models`)

---

## 📋 Project Overview

This project evaluates whether **small language models (SLMs) running locally on consumer hardware** can effectively replace cloud-based AI services. Rather than relying on synthesized benchmarks, we conducted **primary experimental research** comparing three open-source models on 40 original multilingual prompts in Arabic and French.

**Core Research Question:** For practical everyday tasks (factual QA, reasoning, coding, advice), do small models running entirely offline on consumer hardware perform well enough?

---

## 🎯 Key Findings

| Model | Size | Avg Score | Success Rate | Latency | Tokens/sec |
|-------|------|-----------|--------------|---------|-----------|
| **Phi-3 Mini** | 3.8B | **2.03/3** ⭐ | 100% | 34.2s | 7.8 |
| **TinyLlama** | 1.1B | 0.93/3 | 100% | 9.8s | 25.3 |
| **Gemma 2** | 9B | 1.50/3 | 35% | 104.6s | 3.4 |

**Conclusion:** Phi-3 Mini (3.8B) is the practical sweet spot for local multilingual deployment, balancing performance, latency, and reliability on consumer hardware.

---

## 📁 Project Structure

```
virtual_machine/
├── README.md                          # This file
├── prompts.json                       # 40 original test prompts (Arabic/French)
├── requirements.txt                   # Python dependencies
├── install.sh                         # Ollama setup script
│
├── step2_run_experiment.py            # Main evaluation script
├── calculate_stats.py                 # Statistical summary
├── analyze_results.py                 # Detailed analysis & visualization
│
└── experiment_results/                # Generated during experiments
    ├── blind_scoring_20260513_191901.csv     # Raw model responses
    ├── blind_scoring_CLEAN.csv               # Human-scored results
    ├── model_key_20260513_191901.json        # Model anonymization key
    ├── raw_results_20260513_191901.json      # Full JSON results
    ├── failed_prompts_20260513_191901.csv    # Models that timed out
    ├── report_20260513_191901.md             # Full research report
    └── checkpoints/                          # Recovery checkpoints
        ├── checkpoint_tinyllama.json
        ├── checkpoint_phi3_mini.json
        └── checkpoint_gemma2_9b.json
```

---

## 📊 Test Dataset

**40 original multilingual prompts** across 5 categories:

| Category | Examples | Languages |
|----------|----------|-----------|
| **Factual QA** (10) | "What is the capital of Morocco?" | Arabic, French |
| **Reasoning** (10) | Multi-step math, logic puzzles | Arabic, French |
| **Coding** (10) | Python function writing, debugging | Arabic, French |
| **Practical Advice** (10) | Business, health, agriculture tips | Arabic, French |

See [prompts.json](prompts.json) for complete list.

---

## 🚀 Quick Start

### Prerequisites

- Linux/macOS (Ubuntu 22.04+ recommended)
- 16GB+ RAM
- 4+ CPU cores
- ~30GB disk space (for models)
- Internet connection (setup only)

### 1. Install Ollama & Models

```bash
bash install.sh
```

This script:
- Installs Ollama (open-source LLM runtime)
- Pulls three models: TinyLlama, Phi-3 Mini, Gemma 2 9B
- Starts the Ollama service

**First run takes 15-30 minutes** (model downloads: 2GB, 2GB, 6GB)

### 2. Run Experiment

```bash
python3 step2_run_experiment.py
```

This will:
- Execute all 40 prompts on each of 3 models
- Generate blind-scored results (Model A/B/C instead of names)
- Save results to `experiment_results/`
- Create checkpoints after each model

**Estimated runtime:** 2-4 hours (depending on hardware)

### 3. Analyze Results

```bash
python3 analyze_results.py --input experiment_results/blind_scoring_*.csv
```

Generates:
- Descriptive statistics per model
- Cohen's kappa (inter-rater reliability)
- Kruskal-Wallis significance test
- Pairwise Mann-Whitney U comparisons
- Visualizations (boxplots, heatmaps)

---

## 📋 File Descriptions

### Core Scripts

| File | Purpose |
|------|---------|
| `step2_run_experiment.py` | **Main evaluation pipeline** — runs all 40 prompts on each model with checkpoint recovery, timeout handling, and blind anonymization |
| `analyze_results.py` | **Statistical analysis** — inter-rater reliability, non-parametric tests, pairwise comparisons, visualizations |
| `calculate_stats.py` | **Quick summary** — basic descriptive stats grouped by model |

### Data Files

| File | Purpose |
|------|---------|
| `prompts.json` | 40 original test prompts with metadata (category, language, expected type) |
| `requirements.txt` | Python package dependencies |
| `install.sh` | Automated Ollama installation and model setup |

### Results (Generated)

| File | Content |
|------|---------|
| `blind_scoring_*.csv` | Raw model responses with blind anonymization |
| `blind_scoring_CLEAN.csv` | Human-scored responses (0-3 scale) |
| `model_key_*.json` | Maps MODEL_A/B/C to actual model names |
| `raw_results_*.json` | Complete JSON output from all prompts |
| `failed_prompts_*.csv` | Prompts where models timed out or failed |
| `report_*.md` | Formatted markdown report with findings |
| `checkpoints/*.json` | Recovery checkpoints (one per model) |

---

## 🔬 Methodology

### Hardware Configuration

Experiments simulate consumer hardware:
- **CPU:** 4 cores @ 2.4-4.4 GHz
- **RAM:** 16 GB
- **GPU:** None (CPU inference only)
- **Cost:** ~$500 (second-hand laptop, 2022)

### Generation Parameters (Fixed)

All models use identical hyperparameters for fair comparison:

```python
temperature = 0.3      # Deterministic, low randomness
top_p = 0.9           # Nucleus sampling
top_k = 40            # Top-k filtering
num_predict = 256     # Max output tokens
seed = 42             # Reproducibility
```

### Bias Mitigation

- ✅ **Prompt shuffling:** Randomized order to eliminate caching bias
- ✅ **Per-prompt model randomization:** Models execute in random order per prompt
- ✅ **Blind evaluation:** Model names anonymized as MODEL_A/B/C during scoring
- ✅ **Checkpoint recovery:** No data loss if experiment interrupted

---

## 📊 Scoring Rubric

Human evaluators score each response 0–3:

| Score | Criterion | Example |
|-------|-----------|---------|
| **0** | Incorrect/Irrelevant | Q: "Capital of Morocco?" → "Paris" |
| **1** | Partially correct with major errors | Q: "Capital of Morocco?" → "A city in North Africa" |
| **2** | Correct with minor issues | Q: "Capital of Morocco?" → "Rabat (the capital city...)" |
| **3** | Fully correct, clear, complete | Q: "Capital of Morocco?" → "الرباط (Rabat), on the Atlantic coast" |

---

## 💡 Key Insights

### Phi-3 Mini (3.8B) is the "Goldilocks" Model

- **Performance:** Highest average score (2.03/3)
- **Reliability:** 100% success rate, zero timeouts
- **Speed:** Reasonable latency (34.2s average)
- **Efficiency:** 7.8 tokens/sec
- **Verdict:** Production-ready for local multilingual deployment

### TinyLlama (1.1B) is Too Small

- Scored only 0.93/3 despite 100% reliability
- Sub-2B models struggle with non-English tasks
- Fast inference (9.8s) but low quality
- Use only if speed is critical over accuracy

### Gemma 2 (9B) Exceeds Hardware Limits

- Failed on 65% of prompts (14/40 success)
- Memory exhaustion on 16GB RAM
- Practical upper bound for consumer hardware
- Viable only on 32GB+ machines or with quantization

---

## 🌍 Impact & Applications

### Who Can Benefit

- 🚜 **Farmers:** Crop advice, weather patterns, local market info
- 📚 **Students:** Homework help, explanations in native language
- 🏪 **Small businesses:** Customer support, marketing copy, bookkeeping
- 🌐 **Offline communities:** Zero internet required after setup

### Barriers to Adoption

The primary barrier is not technical but **awareness**:
- Most people don't know capable AI runs on their laptops
- Open-source models receive 1% of media coverage vs. ChatGPT
- Perception that "real AI" requires cloud services

---

## 📦 Dependencies

Python packages (see `requirements.txt`):

```
pandas>=2.0.0           # Data manipulation
requests>=2.31.0        # HTTP requests to Ollama
psutil>=5.9.0          # System metrics
numpy>=1.24.0          # Numerical computing
scipy>=1.10.0          # Statistical functions
scikit-learn>=1.3.0    # Machine learning utilities
matplotlib>=3.7.0      # Plotting
seaborn>=0.12.2        # Statistical visualization
statsmodels>=0.14.0    # Statistical models
openpyxl>=3.1.0        # Excel compatibility
```

### External Dependency

- **Ollama:** Open-source LLM runtime (installed by `install.sh`)
  - Download: [ollama.ai](https://ollama.ai)
  - Models pulled: tinyllama, phi3:mini, gemma2:9b

---

## 🔧 Troubleshooting

### Ollama Service Won't Start

```bash
# Check if service is running
pgrep -x "ollama"

# Start manually
ollama serve
```

### Model Download Fails

```bash
# Check disk space
df -h

# Manual model pull
ollama pull tinyllama
ollama pull phi3:mini
ollama pull gemma2:9b
```

### Timeout Errors During Experiment

- Increase `REQUEST_TIMEOUT` in `step2_run_experiment.py` (default: 300s)
- Reduce `OLLAMA_NUM_THREADS` for slower systems
- Run during off-peak hours (fewer background processes)

### Out of Memory

Gemma 2 (9B) may fail on 16GB machines. Options:
1. Use 4-bit quantization (Ollama does this by default)
2. Skip Gemma 2, focus on TinyLlama and Phi-3
3. Upgrade to 32GB RAM

---

## 📖 Full Report

See [experiment_results/report_20260513_191901.md](experiment_results/report_20260513_191901.md) for:
- Complete motivation & background
- Detailed methodology
- Statistical analysis
- Limitations & future work
- Academic citations

---

## 📊 Citation

If you use this work, please cite:

```bibtex
@inproceedings{aamam2026costintelligence,
  title={The Cost of Intelligence: Can Small, Local, and Hybrid AI Replace the Cloud?},
  author={Aamam, Ayman and Abeddad, Fatima Zahra},
  booktitle={Introduction to Data Science and Artificial Intelligence Capstone},
  year={2026}
}
```

---

## 📝 License

This project is open-source. Models are distributed under:
- **TinyLlama:** Apache 2.0
- **Phi-3 Mini:** MIT
- **Gemma 2:** Gemma Terms of Use

See respective model repositories for details.

---

## 🤝 Contributing

Questions or suggestions? Open an issue on [GitHub](https://github.com/Ayman-aa/Capstone-Project).

---

## ✅ Reproducibility

**How to reproduce results:**

1. Clone: `git clone https://github.com/Ayman-aa/Capstone-Project.git`
2. Checkout: `git checkout models`
3. Install: `bash install.sh`
4. Run: `python3 step2_run_experiment.py`
5. Analyze: `python3 analyze_results.py`

**Expected runtime:** ~3 hours on 4-core/16GB hardware

Seed=42 ensures deterministic generation parameters across models.

---

**Last Updated:** May 14, 2026  
**Status:** Complete & Reproducible ✅
