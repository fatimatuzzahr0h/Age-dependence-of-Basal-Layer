"""
Tahap 4.2 - Dosis lapisan robust via integrasi edep silinder
Resep: dose_layer = (sum edep bin dalam jendela kedalaman lapisan) / massa_lapisan.
- Jendela kedalaman per tag dihitung dari zrange ekstraksi v5 vs z_top.
- edep per bin dibagi N_HIST (koreksi bug label Tahap 4.1).
- Konversi klinis: D[Gy] = dose_per_hist * 2 * A[Bq] * t[s].
- Plot PDD dalam PERSEN (normalisasi bin terdangkal) -> bebas ambiguitas massa.
Output: D:\\Brachy\\stage4_output\\robust_clinical_doses.csv + pdd_percent.png
CLI: python 24_stage4b_robust.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROD = Path(r"D:\Brachy\production_output")
WORK = Path(r"D:\Brachy\icrp156_work")
OUT = Path(r"D:\Brachy\stage4_output")
N_HIST = 10_000_000
MEV_TO_J = 1.602176634e-13
BQ_S_PER_HIST = 2.0
ACTIVITY_BQ = 1e9
TIME_S = 60
BIN_MM = 0.1
SPAN_TOP = 2.0      # bin terdangkal berada 2 mm di atas apex
SPAN_DEPTH = 23.0   # kedalaman maksimum grid
TAGS = ["MRCP_01M", "MRCP_01F", "MRCP_05M", "MRCP_05F", "MRCP_15M", "MRCP_15F"]
STANDOFFS = [0.0, 0.5, 1.0]


def read_image(path):
    import itk
    return itk.array_from_image(itk.imread(str(path))).astype(float)


def bin_index(depth_mm):
    return int(round((SPAN_DEPTH - depth_mm) / BIN_MM - 0.5))


def bin_depth(i):
    return SPAN_DEPTH - BIN_MM * (i + 0.5)


def windows(tag, summ):
    r = summ[tag]
    zt = r["z_top_mm"]
    w = {}
    w["epidermis"] = (0.0, zt - r["epidermis"]["zrange_mm"][0])
    w["basal"] = (zt - r["basal"]["zrange_mm"][1], zt - r["basal"]["zrange_mm"][0])
    d_derm = zt - r["epidermis"]["zrange_mm"][0]
    w["dermis"] = (d_derm, d_derm + 2.0)
    w["cranium_cortical"] = (zt - r["cranium_cortical"]["zrange_mm"][1],
                             zt - r["cranium_cortical"]["zrange_mm"][0])
    w["cranium_spongiosa"] = (zt - r["cranium_spongiosa"]["zrange_mm"][1],
                              zt - r["cranium_spongiosa"]["zrange_mm"][0])
    return w


def main():
    OUT.mkdir(exist_ok=True)
    with open(WORK / "extraction_summary_v5_all.json", "r") as f:
        summ = {r["tag"]: r for r in json.load(f)}
    with open(PROD / "production_summary.json", "r") as f:
        prod = json.load(f)
    masses = {}
    for p in (PROD / ".." / "stage23_output" / "stage23_summary.json",):
        pass
    import glob
    st23 = []
    for g in glob.glob(r"D:\Brachy\stage23_output\stage23_summary.json"):
        st23 += json.load(open(g))
    for row in st23:
        masses[(row["tag"], row["layer"])] = row["mass_g"]

    rows = []
    plt.figure(figsize=(10, 6))
    for tag in TAGS:
        w = windows(tag, summ)
        for s in STANDOFFS:
            rec = next(r for r in prod if r["tag"] == tag and r["standoff_mm"] == s)
            edep_total = read_image(PROD / tag / f"s{s:.1f}" / "cyl_edep.mhd").flatten()
            unc = read_image(PROD / tag / f"s{s:.1f}" / "cyl_unc.mhd").flatten()
            edep = edep_total / N_HIST  # MeV per histori
            for L, (d0, d1) in w.items():
                i0, i1 = bin_index(d1), bin_index(d0)
                i0, i1 = max(i0, 0), min(i1, len(edep) - 1)
                sel = slice(i0, i1 + 1)
                e_sum = edep[sel].sum()
                u_rel = float(np.sqrt(np.sum((unc[sel] * edep[sel]) ** 2)) / max(e_sum, 1e-30))
                mass_kg = masses[(tag, L)] / 1000.0
                dose_hist = e_sum * MEV_TO_J / mass_kg
                dose_clin = dose_hist * BQ_S_PER_HIST * ACTIVITY_BQ * TIME_S
                rows.append(dict(tag=tag, standoff_mm=s, layer=L,
                                 depth_window_mm=f"{d0:.2f}-{d1:.2f}",
                                 dose_Gy_per_hist=dose_hist,
                                 dose_Gy_1GBq_60s=dose_clin, unc_rel=u_rel))
        # Kurva PDD persen (s=0) dari edep per bin
        rec0 = next(r for r in prod if r["tag"] == tag and r["standoff_mm"] == 0.0)
        edep0 = read_image(PROD / tag / "s0.0" / "cyl_edep.mhd").flatten() / N_HIST
        depths = np.array([bin_depth(i) for i in range(len(edep0))])
        mask = (depths >= -0.5) & (depths <= 6.0)
        pdd = 100.0 * edep0 / edep0[bin_index(0.05)]
        plt.plot(depths[mask], pdd[mask], label=tag)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "robust_clinical_doses.csv", index=False)
    plt.axhline(100, ls="--", c="k", lw=0.8)
    plt.xlabel("Kedalaman jaringan (mm)")
    plt.ylabel("PDD (%)")
    plt.title("PDD normalisasi permukaan - Sr-90/Y-90, kontak")
    plt.grid(alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "pdd_percent.png", dpi=300)

    print("=" * 78)
    print("TAHAP 4.2 - DOSIS LAPISAN ROBUST (integrasi edep)")
    print("=" * 78)
    piv = df[df.standoff_mm == 0.0].pivot_table(index="tag", columns="layer",
                                                values="dose_Gy_1GBq_60s")
    print(piv.round(3).to_string())
    print("\n--- Standoff (basal) ---")
    print(df[df.layer == "basal"].pivot_table(index="tag", columns="standoff_mm",
                                              values="dose_Gy_1GBq_60s").round(3).to_string())
    print("\nTersimpan:", OUT / "robust_clinical_doses.csv", "|", OUT / "pdd_percent.png")
    print("=" * 78)


if __name__ == "__main__":
    main()