# Age-dependence-of-Basal-Layer
Age-dependence of Basal-layer and Calvaria Marrow Doses in Sr-90/Y-90 Scalp Contact Brachytherapy: A GATE10 Study with ICRP-156 Pediatric Mesh Phantoms
# Layer-resolved Monte Carlo dosimetry of the pediatric scalp for Sr-90/Y-90 contact brachytherapy (ICRP-156 mesh phantoms, GATE 10)

**Version:** 1.0.0 · **Date:** 2026-09-24 · **License:** code MIT / datasets CC-BY-4.0 (see `LICENSE`)
**Status:** analysis complete; manuscript in preparation for *Atom Indonesia*
**Corresponding author:** Fatimatuz Zahroh — fatimatuzzahroh.mp@gmail.com

## 1. Overview

This archive contains the complete computational pipeline, datasets, and figures for:

> *Age dependence of basal-layer and calvarial marrow doses in Sr-90/Y-90 scalp contact brachytherapy: a GATE 10 study with ICRP-156 paediatric mesh phantoms.*

Simulations cover six ICRP-156 paediatric mesh-type reference computational phantoms (1, 5, 15 years; male and female) × three applicator–skin standoffs (0, 0.5, 1.0 mm) = **18 scenarios, 10⁷ histories each**, run with GATE 10.1.1 (Geant4 11.4.2, `G4EmStandardPhysics_option4`).

**Key results (per 1 GBq·60 s, contact geometry):**
- Basal-layer dose 0.69–1.00 Gy; epidermis 1.22–1.63 Gy; basal/epidermis ratio 0.48–0.64, decreasing with age.
- Calvarial spongiosa mid-depth 4.8–5.0 mm (1 y) vs 10.1–10.9 mm (15 y); spongiosa dose 21–27 % of epidermal dose at 1 y vs 1–8 % at 5–15 y (the infant "marrow window").
- A 1.0 mm standoff retains 42–55 % of the contact basal-layer dose.

## 2. Environment

- OS: Windows 10/11; Python 3.11
- `opengate` / `opengate-core` 10.1.1 (statically bundles Geant4 11.4.2)
- `numpy`, `pandas`, `matplotlib`, `itk` (`tabulate` optional)

```bash
pip install opengate==10.1.1 numpy pandas matplotlib itk
```

> Note: on pip/Windows installs Geant4 is bundled inside the `opengate-core` wheel; `geant4-config` is therefore not available.

## 3. Contents

### 3.1 Scripts (pipeline order; entry points per stage)

| Stage | Script | Purpose |
|---|---|---|
| 1 | `04_pdd_validation_coelho.py` | Source-model validation against published depth–dose data |
| 2 | `08_extract_scalp_layers.py` | Vertex scalp extraction (v5) from ICRP-156 meshes |
| 2 | `15_recenter_stl.py` | Re-centring of layers to the vertex origin |
| 2 | `19_place_source_v3.py` | Source placement and gap verification (18 scenarios) |
| 3 | `20_score_mechanism_test.py` | Mechanism test: tessellated vs profile-based scoring |
| 3 | `21_production_pilot.py` | Pilot production (2×10⁶ histories) |
| 3 | `22_batch_production.py` | Production runs (18 × 10⁷ histories, fixed seeds; resume-capable) |
| 4 | `23`–`27` analysis iterations | `26_stage4c_final.py` → `final_layer_doses.csv` (superseded); `27_corrected_depths.py` → **final dataset** |
| 4 | `28_manuscript_assets.py` | Figs. 1–4 and Tables 1–3 for the manuscript |
| 5 | `29_stage5_analysis.py` | Formal analysis 5.1–5.4 (ratios, descriptive stats, standoff, marrow) |
| 6 | `30_pipeline_flowchart.py` | Fig. S1 pipeline flowchart |

Intermediate/fix scripts (09–14, 16–18, 24–25) are retained for provenance only.

### 3.2 Datasets and figures

| File | Status | Description |
|---|---|---|
| `data/manuscript_layer_doses.csv` | **FINAL** | 90 records: layer doses per 1 GBq·60 s + MC uncertainty (k=1) |
| `data/manuscript_tables.md` | **FINAL** | Main-text Tables 1–3 |
| `data/production_summary.json` | **FINAL** | Per-scenario depth–dose profiles (ground truth) + STL-actor doses (known artifact; provenance only) |
| `data/stage25_placement.json` | **FINAL** | Placement verification: gap, penetration, tilt, apex, normal (18 scenarios) |
| `data/stage30_mechanism_test.json` | **FINAL** | Mechanism test (basal/epidermis edep ratio 0.013 vs ≈0.6 expected) |
| `data/stage5_ratios.csv`, `data/stage5_analysis.md` | **FINAL** | Formal analysis outputs (5.1–5.4) |
| `data/pilot_MRCP_01M.json` | provenance | Pilot run records |
| `figures/fig1_pdd.png` … `fig4_marrow_window.png` | **FINAL** | Main-text figures (300 dpi) |
| `figures/pipeline_flowchart.png` | **FINAL** | Fig. S1 |
| `data/clinical_doses.csv` | **OBSOLETE** | ×10⁷ normalization bug — do not cite |
| `data/robust_clinical_doses.csv` | **OBSOLETE** | descending-array `np.interp` bug — do not cite |
| `data/final_layer_doses.csv` | **OBSOLETE** | bounding-box depth error — do not cite |

## 4. Reproduction

1. Obtain the ICRP-156 paediatric MRCPs under licence from the ICRP and place the raw files in `icrp156_raw/` (**not redistributed here**, see §7).
2. Run scripts in numbered order; entry points per stage are listed in §3.1.
3. Production: `python 22_batch_production.py` (resume-capable via per-scenario `done.json` markers). Random seeds are recorded per scenario in `production_summary.json` for exact reproduction.
4. Final dataset: `python 27_corrected_depths.py`; figures/tables: `python 28_manuscript_assets.py`.

Expected runtime: ≈10–20 min per scenario single-threaded (≈3–5 h total for 18 scenarios).

## 5. Methodological notes and known artifacts

- **Thin-shell scoring artifact:** direct scoring on sub-100 µm tessellated shells under-scores by ~2 orders of magnitude (Geant4 navigator tolerance becomes a significant fraction of shell thickness). All reported doses use curvature-corrected profile sampling (manuscript Eqs. 2–3).
- **Normalization:** D_abs = 2·A·t·d_hist, where A is the *nominal ⁹⁰Sr activity* (not the combined beta emission rate); the factor 2 accounts for secular equilibrium (one ⁹⁰Y decay per ⁹⁰Sr decay).
- **Statistics:** no inferential tests are applied — reference phantoms carry no inter-individual variability. Age/sex trends are reported descriptively with Monte Carlo uncertainties (k = 1); every claimed trend exceeds the combined MC uncertainty by ≥ two orders of magnitude.

## 6. Limitations (condensed; full register in manuscript)

L1 operational dermis layer; L2 profile-based scoring recipe; L3 mesh-repair volume losses (≤3.1 %, except 13.6 % for the 15-y male spongiosa — reported as an upper bound); L4 15-y male vertex anomaly handled via local surface normal; L5 input-file conventions verified; L6 composite skin characterization; L7 finite patch (~40 mm) → absolute surface doses a factor 2–4 below a semi-infinite head (ratios and trends robust); L8 curvature-unrolling smearing ±10 % for thin layers.

## 7. Third-party data and licensing

ICRP-156 pediatric mesh phantoms are licensed material of the International Commission on Radiological Protection; **raw meshes are not included** in this archive. Derived STL layers are likewise not redistributed; regeneration scripts are provided so that licensed users can reproduce all geometry locally.

## 8. Declaration of generative AI

Generative AI (Qwen) was used to improve language clarity and organize content during manuscript preparation. All scientific content, data, analyses, and interpretations were produced and verified by the authors, who take full responsibility for the content of this archive and the associated publication.

## 9. Citation

F. Zahroh and F. P. Krisna, 2026. *Age dependence of basal-layer and calvaria marrow doses in Sr-90/Y-90 scalp contact brachytherapy: a GATE 10 study with ICRP-156 pediatric mesh phantoms.* Atom Indonesia (in preparation).
Archive DOI: [10.5281/zenodo.22935992]

## 10. Contact

Fatimatuz Zahroh, Gatoel Hospital, fatimatuzzahroh.mp@gmail.com
