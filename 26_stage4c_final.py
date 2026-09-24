"""
Tahap 4.2 v3 (final) - Dosis lapisan robust, dua bug diperbaiki:
(1) profil di-sort NAIK menurut kedalaman sebelum np.interp;
(2) profil dibagi N_HIST (cyl_dose.mhd = dosis total run).
Dosis lapisan = rata-rata berbobot cincin atas profil:
  d(r) = d_apex + r^2/(2R); w = r; dose = sum(P(d(r))*r)/sum(r)
Konversi klinis: D[Gy] = dose[Gy/hist] * 2 * A[Bq] * t[s].
Output: final_layer_doses.csv + pdd_percent.png (diperbarui)
CLI: python 26_stage4c_final.py
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
ACTIVITY_BQ = 1e9
TIME_S = 60
BIN_MM = 0.1
SPAN_DEPTH = 23.0
TAGS = ["MRCP_01M", "MRCP_01F", "MRCP_05M", "MRCP_05F", "MRCP_15M", "MRCP_15F"]
STANDOFFS = [0.0, 0.5, 1.0]
LAYERS = ["epidermis", "basal", "dermis", "cranium_cortical", "cranium_spongiosa"]
T_BASAL = {"MRCP_01M": 0.0616, "MRCP_01F": 0.0616, "MRCP_05M": 0.0627,
           "MRCP_05F": 0.0600, "MRCP_15M": 0.0519, "MRCP_15F": 0.0511}


def read_image(p):
    import itk
    return itk.array_from_image(itk.imread(str(p))).astype(float)


def bin_depth(i):
    return SPAN_DEPTH - BIN_MM * (i + 0.5)


def load_profile(tag, s):
    raw = read_image(PROD / tag / f"s{s:.1f}" / "cyl_dose.mhd").flatten() / N_HIST
    unc = read_image(PROD / tag / f"s{s:.1f}" / "cyl_unc.mhd").flatten()
    dep = np.array([bin_depth(i) for i in range(len(raw))])
    ok = (dep >= 0.0) & (dep <= 22.0)
    dep, raw, unc = dep[ok], raw[ok], unc[ok]
    order = np.argsort(dep)          # PERBAIKAN BUG 1: urutkan menaik
    return dep[order], raw[order], unc[order]


def main():
    OUT.mkdir(exist_ok=True)
    summ = {r["tag"]: r for r in json.load(open(WORK / "extraction_summary_v5_all.json"))}
    st23 = json.load(open(r"D:\Brachy\stage23_output\stage23_summary.json"))
    mass = {(r["tag"], r["layer"]): r["mass_g"] for r in st23}
    dens_skin = {r["tag"]: r["density_gcm3"] for r in st23 if r["layer"] == "epidermis"}
    for tag, r in summ.items():
        mass[(tag, "dermis")] = r["dermis"]["volume_mm3"] * dens_skin[tag] / 1000.0

    rows = []
    plt.figure(figsize=(10, 6))
    for tag in TAGS:
        r = summ[tag]
        zt = r["z_top_mm"]
        sag = r["basal"]["zrange_mm"][1] - r["basal"]["zrange_mm"][0]
        area = r["basal"]["volume_mm3"] / T_BASAL[tag]
        r_max = np.sqrt(area / np.pi)
        R = r_max ** 2 / (2.0 * sag) if sag > 0 else 100.0
        rr = np.linspace(0.0, r_max, 60)
        w = rr.copy()

        for s in STANDOFFS:
            dep, prof, uncp = load_profile(tag, s)
            for L in LAYERS:
                vol = r[L]["volume_mm3"]
                t_l = vol / area
                d_apex = (zt - r[L]["zrange_mm"][1]) + t_l / 2.0
                d_r = d_apex + rr ** 2 / (2.0 * R)
                p_r = np.interp(d_r, dep, prof)
                dose_hist = float(np.sum(p_r * w) / np.sum(w))
                dose_apex = float(np.interp(d_apex, dep, prof))
                unc = float(np.interp(d_apex, dep, uncp))
                dose_clin = dose_hist * 2.0 * ACTIVITY_BQ * TIME_S
                rows.append(dict(tag=tag, standoff_mm=s, layer=L,
                                 d_apex_mm=round(d_apex, 3),
                                 dose_Gy_per_hist=dose_hist,
                                 dose_apex_Gy_per_hist=dose_apex,
                                 dose_Gy_1GBq_60s=dose_clin,
                                 unc_rel=unc, mass_g=mass[(tag, L)]))

        dep0, prof0, _ = load_profile(tag, 0.0)
        m0 = dep0 <= 6.0
        plt.plot(dep0[m0], 100.0 * prof0[m0] / prof0[np.argmin(abs(dep0 - 0.05))], label=tag)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "final_layer_doses.csv", index=False)
    plt.axhline(100, ls="--", c="k", lw=0.8)
    plt.xlabel("Kedalaman jaringan (mm)")
    plt.ylabel("PDD (%)")
    plt.title("PDD normalisasi permukaan - Sr-90/Y-90, kontak")
    plt.grid(alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "pdd_percent.png", dpi=300)

    print("=" * 78)
    print("TAHAP 4.2 v3 - DOSIS LAPISAN FINAL ( Gy per 1 GBq x 60 s )")
    print("=" * 78)
    piv = df[df.standoff_mm == 0.0].pivot_table(index="tag", columns="layer",
                                                values="dose_Gy_1GBq_60s")
    print(piv.round(3).to_string())
    print("\n--- Rasio basal/epidermis (kontak) ---")
    e = df[(df.standoff_mm == 0.0) & (df.layer == "epidermis")].set_index("tag")["dose_Gy_1GBq_60s"]
    b = df[(df.standoff_mm == 0.0) & (df.layer == "basal")].set_index("tag")["dose_Gy_1GBq_60s"]
    print((b / e).round(3).to_string())
    print("\n--- Efek standoff: basal & epidermis ---")
    for L in ["epidermis", "basal"]:
        print(f"[{L}]")
        print(df[df.layer == L].pivot_table(index="tag", columns="standoff_mm",
                                            values="dose_Gy_1GBq_60s").round(3).to_string())
    print("\nTersimpan:", OUT / "final_layer_doses.csv")
    print("=" * 78)


if __name__ == "__main__":
    main()