# igd-alphaearth

Code, data products and extended results for the IGD 2026 paper
**Sample Efficiency of AlphaEarth Embeddings for Land Cover Classification: Spatial Holdout and Cross-Country Evaluation**
(11th International Symposium Intercontinental Geoinformation Days, Alanya, Türkiye, 16–17 December 2026).

Author: Abdüsselam Kesikoğlu, Tomarza MA Vocational School, Kayseri University, Türkiye ([ORCID 0000-0003-4976-0791](https://orcid.org/0000-0003-4976-0791), kesikoglu@kayseri.edu.tr)

The repository reproduces every number, table and figure of the paper from openly available Google Earth Engine data.
Detailed results that are summarised in the paper (scores for all feature sets and both classifiers, sensitivity
analyses, confusion matrices, Sentinel-1 scene counts, dissimilarity distributions, algorithms) are given in
[`docs/Extended_Results.pdf`](docs/Extended_Results.pdf).

## Study design (frozen in `config/config.yaml` before any model was trained)
- **Reference data:** LUCAS 2018 Copernicus polygons (`JRC/LUCAS_HARMO/COPERNICUS_POLYGONS/V1/2018`), screened subset, level one groups B to F, area ≥ 400 m²; 10,144 polygons (Spain 8,359, Greece 1,785).
- **Features (polygon means, 10 m nominal grid, cells weighted by covered area fraction):** AlphaEarth Satellite Embedding 2018 (64); Sentinel-2 L2A 2018 masked with Cloud Score+ (cs_cdf ≥ 0.60), annual (10) and quarterly (40) medians; Sentinel-1 GRD quarterly VV/VH medians added to S2 seasonal (48; also ascending-only and descending-only); AE + S2 seasonal (104); AE rescaled to unit length.
- **Classifiers:** Random Forest (300 trees) and multinomial logistic regression, fixed hyperparameters (scikit-learn 1.9.1).
- **Training budgets:** 10, 25, 50 and 100 labels per class × 20 draws (identical polygon IDs for all feature sets) + full pool of 6,129 polygons.
- **Testing:** locked holdout of 62 of 248 Spanish blocks of 50 km (EPSG:3035) with a 5 km buffer; external test in Greece; validation sensitivity analysis with random and 10 to 100 km block holdouts.
- **Uncertainty:** two-level cluster bootstrap (2,000 draws) over test blocks and training draws, paired across representations.

## Repository layout
| Path | Content |
|---|---|
| `config/config.yaml` | Frozen analysis settings |
| `src/` | All scripts (see pipeline below) |
| `data/raw/` | Reference table and polygon features extracted from Earth Engine; national boundaries (LSIB 2017) |
| `data/processed/samples.parquet` | Analysis table (10,144 polygons × all features) |
| `data/splits/` | Identifier lists of training pool, locked test, buffer zone and Greek test with SHA-256 manifest; 50 km block polygons (GeoJSON) |
| `data/manifests/datasets.json` | Asset IDs, extraction date, reducer, checksum |
| `outputs/metrics/` | Per run scores (`e1_runs.csv`, `e2_ladder.csv`), summaries, paired differences with intervals, per sample predictions (`predictions_long.parquet`) |
| `outputs/tables/`, `outputs/figures/` | Paper tables (CSV) and figures (600 dpi PNG and vector PDF), extended tables and figures |
| `docs/Extended_Results.pdf` | Extended results and processing details |
| `reports/` | Selection audit, dataset audit, split audit, Sentinel-1 scene counts, DOI check, text audit, environment lock |
| `manuscript/` | Source text (`manuscript.md`) and reference list used to build the paper |

## Reproduce
```bash
git clone https://github.com/a-kesikoglu/igd-alphaearth.git
cd igd-alphaearth
pip install -r reports/requirements-lock.txt     # Python 3.12
python reproduce_all.py                          # offline: from archived feature tables to all tables and figures
python reproduce_all.py --extract                # optionally re-extract features (needs: earthengine authenticate)
```
Earth Engine extraction requires an Earth Engine enabled Google Cloud project; set `ee_project` in `config/config.yaml`.
Building the paper DOCX/PDF (`src/06_build_manuscript.py`, `src/07_render.py`) additionally needs the IGD 2026 Word
template and Microsoft Word; all other steps run on any platform.

| Step | Script | Main output |
|---|---|---|
| Earth Engine extraction | `01_export_features.py`, `01b_boundaries.py`, `01c_extras.py`, `01d_qc_s1.py` | `data/raw/`, `reports/selection_audit.csv`, `reports/s1_scene_counts.csv` |
| Quality control | `02_prepare.py` | `data/processed/samples.parquet`, `reports/dataset_audit.md` |
| Experiments | `03_experiments.py` | `outputs/metrics/e1_*`, `e2_ladder.csv`, `data/splits/` |
| Statistics and tables | `04_statistics.py`, `04b_confusion.py`, `04c_tables.py`, `04d_shift.py` | `outputs/metrics/`, `outputs/tables/` |
| Figures | `05_figures.py`, `05_figures_results.py`, `05b_workflow_figure.py` | `outputs/figures/` |
| Predictions and extended results | `09_export_predictions.py`, `10_supplement.py` | `outputs/metrics/predictions_long.parquet`, `docs/Extended_Results.docx` |
| Text audit | `08_quality_check.py` | `reports/manuscript_audit.md` |

### Prediction table columns
`run_id, test_set, sample_id, point_id, block_id, true_label, prediction, representation, classifier, budget, repetition, rf_seed, sampling_seed`

## Data licences and credits
- LUCAS 2018 Copernicus module: European Commission (Eurostat, Joint Research Centre); reuse with attribution.
- AlphaEarth Foundations Satellite Embedding dataset: produced by Google and Google DeepMind (CC BY 4.0).
- Sentinel-1 and Sentinel-2: European Union Copernicus programme. Cloud Score+: Google.
- National boundaries: US Department of State, LSIB 2017 (public domain).
- Code: MIT License (see `LICENSE`).

## Citation
Kesikoğlu, A. (2026). Sample efficiency of AlphaEarth embeddings for land cover classification: Spatial holdout and cross-country evaluation. In *Proceedings of the 11th International Symposium Intercontinental Geoinformation Days (IGD 2026)*. Alanya, Türkiye. (Page numbers to be added after publication.)
