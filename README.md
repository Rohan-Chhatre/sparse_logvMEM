# GPU-accelerated Regularized Vector Multiplicative Error Models for Positive-valued Financial Time Series

This repository contains R code and selected computational outputs accompanying the manuscript **“GPU-accelerated Regularized Vector Multiplicative Error Models for Positive-valued Financial Time Series”**, prepared for submission to *Computational Statistics*.

**Authors (manuscript order):** Rohan Hemant Chhatre, Chiranjit Dutta, Sumanta Basu, and Nalini Ravishanker.

## Overview

The paper studies sparse logarithmic vector multiplicative error models (log-vMEMs) with multivariate Gamma innovations. The proposed framework combines hierarchical lag (HLag) structures—componentwise, own-other, and elementwise—with group lasso, adaptive group lasso, group MCP, and group SCAD penalties. Estimation uses alternating block-coordinate updates and a corrected common-factor multivariate Gamma likelihood. GPU-based numerical quadrature is used to accelerate repeated likelihood evaluations.

## Repository structure

- [`codes/`](codes/) — R functions and scripts for the model, estimation routines, penalties, and CPU/GPU likelihood evaluations.
- [`simulation_study/`](simulation_study/) — materials relating to the simulation analyses in the manuscript.
- [`data/`](data/) — data materials included with the repository. See **Data provenance and availability** below.
- [`real_data_analysis/rolling_GPU/final_rolling_tables/`](real_data_analysis/rolling_GPU/final_rolling_tables/) — selected reported output tables from the GPU rolling-window analysis.

## Computational workflow

The study estimates unpenalized and penalized log-vMEMs, performs simulation-based parameter and structure recovery assessments, and evaluates five realized-volatility measures constructed from Microsoft (MSFT) intraday transactions. The rolling-window assessment covers 27 forecast origins and compares forecast horizons 1–5.

The CPU and GPU programs serve different computational purposes. GPU scripts rely on a CUDA-compatible environment and the R `torch` interface; the rolling workflow also uses SLURM array-task configuration. Full rolling fits are computationally intensive. The repository is intended to provide code and supporting materials, **not** to imply that a complete multi-origin refit can be performed with a single command on an ordinary laptop.

## Data provenance and availability

The empirical study uses five daily realized-volatility measures derived from MSFT transaction-level data covering 2010–2015. The `data/` directory contains the preprocessed data used in this study.

## Reproducibility and numerical notes

The likelihood uses a common-factor multivariate Gamma innovation model; the model-implied off-diagonal **error correlations** are fixed at 0.5 by parameterization. GPU timing results for isolated likelihood evaluations should be distinguished from end-to-end estimation time. The full CPU rolling-run runtime stated in the paper is an extrapolation, not a measured 27-origin CPU run. The reported results refer to the specific optimization and quadrature settings documented in the paper and scripts.

## Contact

**Rohan Hemant Chhatre**  
Department of Statistics, University of Connecticut  
Email: rohan.chhatre@uconn.edu

## Acknowledgements

The authors thank the University of Connecticut Storrs HPC support staff for maintaining the computational resources used in this research.

## Citation

Until this work is published in a peer reviewed journal, please cite the pre-print of this manuscript:

@misc{chhatre2026gpuacceleratedregularizedvectormultiplicative,
      title={GPU-accelerated Regularized Vector Multiplicative Error Models for Positive-valued Financial Time Series}, 
      author={Rohan Hemant Chhatre and Chiranjit Dutta and Nalini Ravishanker and Sumanta Basu},
      year={2026},
      eprint={2610.08443},
      archivePrefix={arXiv},
      primaryClass={stat.CO},
      url={https://arxiv.org/abs/2610.08443}, 
}
We request you to cite the associated manuscript when using this code. The final bibliographic reference and DOI can be added following publication.
