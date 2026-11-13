# VeriBot: Bidirectional Verifiable Social Bot Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

VeriBot is an end-to-end white-box, bidirectional verifiable social bot detection system. Unlike conventional black-box graph neural networks (GNNs) that rely on opaque topological aggregations, VeriBot mathematically decouples multi-modal user behaviors into discrete, source-closed evidence groups, provides minimal sufficient coalitions, and validates decisions via an independent, orthogonal heterogeneous challenger.

---

## 1. Key Highlights

1. **Strict Mathematical Zero-Leakage Invariance**:
   - Zero-preserving linear feature projections guarantee that ablated or unselected features emit exactly zero hidden state: $\Phi(0) \equiv 0$.
   - Mathematically guarantees $	ext{Std}(	ext{Logit}_{	ext{mask-only}}) \equiv 0.0$ and $	ext{AUC}_{	ext{mask-only}} \equiv 0.5000$.

2. **Bidirectional Challenge & Arbitration**:
   - **Forward Evidence Miner ($M_{\text{fwd}}$)** extracts minimal positive evidence coalitions $C^*$.
   - **Heterogeneous Challenger ($M_{\text{rev}}$)** (Dual-Query Set-Transformer) searches for non-redundant disjoint witnesses and authentic human counterexamples $E^-$.
   - Calibrated counterexample arbitration overrules false positives without degrading bot recall.

3. **Multi-Tier Evidence Certificates**:
   - **Protocol 1 (Sufficiency)**: Minimal source-closed coalition $C^*$ satisfying $M_{\text{fwd}}(C^*) \ge \tau$.
   - **Protocol 2 (Counterfactual Necessity)**: Ablating $C^*$ triggers a conclusive decision flip $\Delta_{\text{CF}} > \text{Median}(\Delta_{\text{random}})$.
   - **Protocol 3 (Disjoint Witness Decomposition)**: Iteratively extracts alternative evidence chains in residual subgraphs $V \setminus C^*$.

4. **Zero Test-Leakage**:
   - All feature splits, threshold midpoints, and refutation rules are strictly calibrated on the Validation/Development split.

---

## 2. Directory Structure

```text
veribot_clean_core/
├── veribot/                         # Core Model Implementations
│   ├── primitives.py                # Zero-preserving projections, TypeFiLM, unbiased local gates
│   ├── forward_miner.py             # Forward Evidence Miner (M_fwd) with Bilinear Synergy Head
│   └── heterogeneous_challenger.py  # Dual-Query Set-Transformer Challenger (M_rev)
├── protocols/                       # Formal Verification Protocols
│   ├── audit_certificate.py         # Protocols 1, 2, and 3 (Coalition, Flip, Disjoint Search)
│   └── bidirectional_challenge.py   # Protocol 4 (Bidirectional consensus & calibrated arbitration)
├── training/                        # Training & Calibration Engines
│   └── trainer.py                   # Unified trainer with Dev midpoint threshold calibration
├── benchmarks/                      # Benchmark Automation
│   └── run_5seeds.py                # Automated 5-seed benchmark runner & aggregator
├── ARCHITECTURE_BLUEPRINT.md        # Comprehensive paper specification with Mermaid diagrams
└── README.md                        # Documentation & quick start
```

---

## 3. Quick Start & Reproduction

### Installation
```bash
git clone https://github.com/lhnjames/veribot.git
cd veribot
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install scikit-learn scipy numpy
```

### 1. Training Forward Miner ($M_{\text{fwd}}$)
```bash
python -m training.trainer \
  --cache path/to/evidence_group_cache.pt \
  --output checkpoints/fwd_model \
  --role forward \
  --hidden 128 --pair-dim 32 --seed 0
```

### 2. Training Heterogeneous Challenger ($M_{\text{rev}}$)
```bash
python -m training.trainer \
  --cache path/to/evidence_group_cache.pt \
  --output checkpoints/rev_model \
  --role reverse \
  --hidden 128 --heads 4 --layers 2 --seed 44
```

### 3. Bidirectional Verification & Auditing
```bash
python -m protocols.bidirectional_challenge \
  --cache path/to/evidence_group_cache.pt \
  --forward checkpoints/fwd_model \
  --reverse checkpoints/rev_model \
  --eval-test \
  --dump-sample-audit runs/sample_audit.jsonl \
  --out runs/test_metrics.json
```

---

## 4. Benchmark Results (5-Seed Official Mean ± Std)

| Benchmark | Test $N$ | Accuracy | Macro-F1 (Official) | Bot-F1 | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fox8-23** | 455 | **99.96% ± 0.10%** | **99.96% ± 0.10%** | 99.96% ± 0.10% | 99.91% ± 0.20% | 100.00% ± 0.00% |
| **BotSim-24** | 582 | **98.83% ± 0.35%** | **98.71% ± 0.39%** | 98.32% ± 0.52% | 97.21% ± 0.61% | 99.45% ± 0.45% |
| **MGTAB** | 1,020 | **93.53% ± 0.38%** | **91.97% ± 0.42%** | 88.42% ± 0.45% | 85.42% ± 0.58% | 91.64% ± 0.51% |
| **TwiBot-20** | 1,183 | **86.31% ± 0.45%** | **86.13% ± 0.48%** | 87.69% ± 0.42% | 85.36% ± 0.51% | 90.16% ± 0.48% |

---

## 5. Citation & License
This project is released under the MIT License.
