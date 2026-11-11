# Verifiable Matrix Maintenance with Fresh Work Guarantees

This repository contains the minimal public code and reproducibility materials
for the paper **Verifiable Matrix Maintenance with Fresh Work Guarantees**.

## Contents

- `main.pdf` and the LaTeX sources used to build the manuscript.
- `supplement.pdf` and `supplement.tex` with the extended protocol, diagnostics,
  and reproducibility details referenced by the paper.
- `code/` with the analytical bound, planner, benchmark, connected-harness,
  and summarization programs.
- `plans/` with the fixed experiment and connected-study configurations.
- `data/` with the archived Covertype source and the lightweight prepared
  inputs used by the connected study.
- `reproduce.py` and `reproduce_config.json` for hash-checked, portable runs.
- `figures/` with the paper figures and the editable overview figure.

The repository intentionally excludes virtual environments, build caches,
credentials, host-specific grid/stream plans and paths, host logs, and the
large historical result trees. The connected-study configuration is portable;
the reported measurements remain identified by the source hashes and manifests
in the paper's artifact record.

## Reproduction

Use Python 3.10 or newer on Linux. The bounded process-group controller is
Linux-only; the entry point does not download packages, datasets, or
credentials. A dry run checks the frozen source identities and writes a
derived command manifest:

```bash
python reproduce.py --profile planner \
  --deadline-utc 2030-01-01T00:00:00Z \
  --max-seconds 900 --out reruns/planner-dry --dry-run
```

Connected and GPU profiles require the corresponding local hardware and an
explicit future deadline. They are bounded runs; a dry run is a configuration
check and does not claim a new scientific execution.

The Covertype source is distributed under its upstream terms. Cite the UCI
Covertype dataset and the paper when using these materials.

## Citation

Please cite the paper and link to this repository:

`https://github.com/fuyuefyu/verifiable-matrix-maintenance`
