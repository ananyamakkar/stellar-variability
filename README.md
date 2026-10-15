# Stellar Photometric Variability: A Statistical Investigation

**Question:** What stellar properties and characteristics are statistically associated with the size and nature of stellar photometric variability? All results are associations in catalogued samples, not causal claims.

**Author:** Ananya Makkar 

---

## What is in this repository

| Folder / file | Contents |
|---|---|
| `notebooks/` | One notebook per stage (see below) |
| `src/stats_toolkit.py` | Statistical functions coded from the formulas, each verified against scipy or statsmodels in the notebooks |
| `data/processed/` | Cleaned and derived tables produced by the notebooks |
| `data/raw/` | Raw downloads (**not in the repository**, see "Reproducing the data") |
| `requirements.txt` | Pinned Python packages |

## Notebooks

| Notebook | Topic |
|---|---|
| `01_pilot_gaia` | Gaia DR3 pilot pull, class counts, missing values, amplitude and period columns |
| `02_period_amplitude_relationship` | RR Lyrae and Cepheid period vs amplitude: Pearson, Spearman, Kendall, regression, curve fitting, t, Z and F tests, power simulation (Modules 1 to 4) |
| `03_sample_design` | Reproducible Gaia sample design (`random_index`), confidence and brightness cuts |
| `04_class_comparison` | Amplitude by variability class: one-way and two-way ANOVA, Welch, Kruskal-Wallis, Games-Howell, chi-square tests |
| `05_derived_vs_catalogued` | AAVSO light curve of delta Cep: derived vs catalogued amplitude |
| `06_cepheid_sample_comparison` | Six classical Cepheids: derived (Fourier) vs VSX amplitude, paired t-test, Bland-Altman |
| `07_extensions` | Reproducible RR Lyrae sample, partial correlation, multiple regression with VIF, interaction model, one-sample t and Z tests |
| `08_stellar_properties` | Scatter diagrams by class and a pilot of Gaia stellar parameters |

## Statistical toolkit

`src/stats_toolkit.py` implements, from the formulas: Pearson r (with SE, probable error and Fisher-z interval), Spearman, Kendall tau-b, least-squares lines and polynomials, one- and two-sample t, one- and two-sample Z, two-proportion Z, F-test for variances, regression F-test, one-way and two-way ANOVA, Games-Howell, chi-square (independence, goodness of fit, single variance), Fourier-series fitting, partial correlation and VIF.

## Data sources

- **Gaia DR3** (ESA): variability tables and `gaia_source`, queried through astroquery on 2026-10-12 to 2026-10-15.
- **AAVSO VSX**: catalogued periods and magnitude ranges for classical Cepheids (export of 2026-10-14).
- **AAVSO International Database**: Johnson V observations, 2024-01-01 to 2025-12-31 (downloaded 2026-10-13 to 2026-10-15).

Gaia data are used under ESA's data-use terms. AAVSO data belong to AAVSO and its observers. See their websites for terms and acknowledgements.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Developed with Python 3.14 in VS Code with Jupyter. Run notebooks from the `notebooks/` folder, because paths are relative (`../data`, `../src`).

## Reproducing the data

Raw files are not stored here. To rerun the notebooks, recreate `data/raw/` as follows:

1. **Gaia files:** run the queries in notebooks `01_pilot_gaia` and `03_sample_design`. The main sample uses `random_index < 20,000,000` ordered by `random_index`, so the same query returns the same stars.
2. **VSX file:** `vsx_dcep_search_20261014.csv`, exported from the VSX search page (type DCEP, period 3 to 15 d, V brighter than 8).
3. **AAVSO files:** one CSV per star in `data/raw/aavso_cepheids/`, named `AUID_Name.csv`. The AUIDs and row counts are listed in `data/processed/cepheid_targets_counts.csv`. Settings: Johnson V, 2024-01-01 to 2025-12-31, no observer filter, comments included.
4. **delta Cep for notebook 05:** `observations_20261013_101102.csv` (480 rows, same settings).

Archive contents change over time, so row counts may differ slightly if you download later.

## Main findings

- **Period vs amplitude (RR Lyrae, n = 2000):** pooled r = +0.41, but fundamental mode r = -0.41 and overtone r = +0.22. The pooled sign is the opposite of the fundamental-mode sign (Simpson's paradox). Mode explains far more amplitude variation than period, colour or brightness (R² 0.17 for period alone within the fundamental mode, 0.45 once mode is added).
- **Class vs amplitude (2079 stars, 8 classes):** one-way ANOVA F(7, 2071) = 1450.6, eta² = 0.83. Welch ANOVA and Kruskal-Wallis agree. Class and colour are strongly associated (Cramer's V = 0.73).
- **Derived vs catalogued amplitude (6 Cepheids):** derived amplitudes are on average 0.07 mag below VSX (95% CI -0.153 to 0.013, p = 0.083). With n = 6 the test has low power. This is a small sample of bright, well-observed stars.

## Limitations

- Catalogued samples over-represent bright, large-amplitude stars. Class labels and periods come from Gaia's pipeline.
- Stellar temperature, luminosity and radius from Gaia were mostly missing or implausible for RR Lyrae in a pilot, so colour and G magnitude stand in for stellar properties.
- The Cepheid comparison uses six stars after selection rules (23 candidates, 8 with enough Johnson V rows, 6 with usable phase coverage). Bootstrap intervals ignore period uncertainty and observer offsets.
- Several planned items were not done: spectral-type groups, a Gaia-VSX cross-match, period estimation from light curves, and a machine-learning notebook.

## Licence

All rights reserved. This project is not licensed for reuse or redistribution.