# 🎓 Invigilator Assignment Problem (IAP) Solver

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PuLP](https://img.shields.io/badge/PuLP-Optimization-orange.svg)](https://coin-or.github.io/pulp/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-green.svg)](https://pandas.pydata.org/)

An optimization system that automates exam invigilator scheduling for universities. The project applies **Mixed-Integer Linear Programming (MILP)** combined with **Goal Programming** to produce assignment schedules that are not only feasible but also fair and humane for the teaching staff.

## 📐 System Architecture

```
IAPDataset.csv ──▶ dataprocessing.py ──▶ processed_data.json ──▶ solver.py ──▶ Output.csv
                                                                     │
                                                                     ▼
                                                          Performance Analysis
```

## 🌟 Key Features

The system is organized around two sets of constraints that comprehensively address real-world university scheduling challenges:

### 🔒 Hard Constraints

Mandatory constraints — any violation renders the solution infeasible:

| # | Constraint | Description |
|---|-----------|-------------|
| H1 | **Staffing Requirements** | Each shift is assigned exactly the required number of staff, with campus-specific roles (LTK or DiAn) |
| H2 | **Clash-Free** | An invigilator cannot be assigned to two shifts at the same time on the same day |
| H3 | **No Cross-Campus Travel** | An invigilator cannot be assigned to consecutive shifts on different campuses |
| H4 | **Single Role per Shift** | Each invigilator holds at most one role per shift |

### 🎯 Soft Constraints (Goal Programming)

Optimizes staff comfort through weighted penalty scores:

| # | Constraint | Weight | Description |
|---|-----------|--------|-------------|
| S1 | **Workload Fairness** | `w = 15` | Balances workload so that every invigilator's shift count is close to the average |
| S2 | **Fatigue Minimization** | `w = 5` | Minimizes back-to-back shift assignments to reduce fatigue |
| S3 | **Campus Affinity** | `w = 2` | Reduces assignments to non-preferred campuses based on staff location preference |

**Objective Function (Minimize):**

```
Z = 15 × Workload_Penalty + 5 × Fatigue_Penalty + 2 × Travel_Penalty
```

## 📁 Project Structure

```
├── IAPDataset.csv          # Input data (original exam schedule / baseline)
├── dataprocessing.py       # Data preprocessing: CSV → JSON
├── processed_data.json     # Processed data (sets & parameters)
├── solver.py               # MILP model + Performance Analysis
├── Output.csv              # Optimized assignment output
├── test_file.py            # Data integrity validation
├── hard.py                 # Hard constraint prototype (reference)
├── requirement.txt         # Python dependencies
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+

### Installation

```bash
pip install -r requirement.txt
```

### Usage

```bash
# Step 1: Preprocess the input data
python dataprocessing.py

# Step 2 (optional): Validate the processed data
python test_file.py

# Step 3: Run the solver + performance analysis
python solver.py
```

The optimized schedule is exported to `Output.csv`. A Baseline vs. Optimized performance comparison table is printed directly in the terminal.

## 📊 Performance Results

| Metric | Baseline | Optimized |
|--------|----------|-----------|
| Total assigned duties | 769 | 769 |
| Max workload per invigilator | 19 | 11 |
| Min workload per invigilator | 3 | 10 |
| MAD from ideal workload | 1.5808 | 0.4977 |
| Simultaneous-shift conflicts | 0 | 0 |
| Cross-campus consecutive conflicts | 4 | 0 |
| Total fatigue penalty | 271 | 0 |
| Total travel preference penalty | 248 | 106 |

## 🛠️ Tech Stack

- **[PuLP](https://coin-or.github.io/pulp/)** — LP/MILP modeling library for Python
- **[Pandas](https://pandas.pydata.org/)** — Data processing and analysis
- **[NumPy](https://numpy.org/)** — Numerical computation
