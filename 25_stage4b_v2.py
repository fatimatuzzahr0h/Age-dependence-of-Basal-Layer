"""
Tahap 4.2 v2 - Dosis lapisan robust dengan koreksi kurvatur (ring-weighted)
Menggantikan 24_stage4b_robust.py (KeyError dermis + window planar tidak sah
untuk shell melengkung).
Resep:
- Massa dermis disintesis: vol_ekstraksi_v5 x densitas skin (layer operasional).
- Dosis lapisan = rata-rata berbobot cincin dari profil silinder:
    d(r) = d_apex + r^2/(2R);  w = r;  dose = sum(profile(d(r))*r)/sum(r)
  d_apex = (z_top - zmax_layer) + t_layer/2 ; t_layer = vol_layer/area_patch
  area_patch = vol_basal/t_basal ; R = r_max^2/(2*sag_basal)
- Referensi pembanding: dosis apex (r=0) pada d_apex.
- Konversi klinis: D[Gy] = dose[Gy/hist] * 2 * A[Bq] * t[s]  (setara sekuler).
Output: robust_clinical_doses.csv + pdd_percent.png + tabel pivot konsol.
CLI: python 25_stage4b_v2.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROD = Path(r"D:\Brachy\production_output")
WORK = Path(r"D:\Brachy\icrp156_work")
ST23 = Path(r"D:\Brachy\stage23_output\stage23_summary.json")
OUT = Path(r"D:\Brachy\stage4_output")

N_HIST = 10_000_000
ACTIVITY_BQ = 1e9
TIME_S = 60
BIN_MM = 0.1
SPAN_DEPTH = 23.0
TAGS = ["MRCP_01M", "MRCP_01F", "MRCP_05M", "MRCP_05F", "MRCP_15M", "MRCP_15F"]
STANDOFFS = [0.0, 0.5, 1.0]
LAYERS = ["epidermis", "basal", "dermis", "cranium_cortical", "cranium_spongiosa"]
# Tebal basal terukur (Tahap 2.4, mm) untuk estimasi area patch
T_BASAL = {"MRCP_01M": 0.0616, "MRCP_01F": 0.0616, "MRCP_05M": 0.0627,
           "MRCP_05F": 0.0600, "MRCP_15M": 0.0519, "MRCP_15F": 0.0511}


def read_image(p):
    import itk
    return itk.array_from_image(itk.imread(str(p))).astype(float)


def bin_depth(i):
    return SPAN_DEPTH - BIN_MM * (i + 0.5)


def main():
    OUT.mkdir(exist_ok=True)
    summ = {r["tag"]: r for r in json.load(open(WORK / "extraction_summary_v5_all.json"))}
    prod = json.load(open(PROD / "production_summary.json"))
    st23 = json.load(open(ST23))

    mass = {(r["tag"], r["layer"]): r["mass_g"] for r in st23}
    dens_skin = {r["tag"]: r["density_gcm3"] for r in st23 if r["layer"] == "epidermis"}
    for tag, r in summ.items():
        mass[(tag, "dermis")] = r["dermis"]["volume_mm3"] * dens_skin[tag] / 1000.0

    rows = []
    plt.figure(figsize=(10, 6))
    for tag in TAGS:
        r = summ[tag]
        zt = r["z_top_mm"]
        vb = r["basal"]["volume_mm3"]
        sag = r["basal"]["zrange_mm"][1] - r["basal"]["zrange_mm"][0]
        area = vb / T_BASAL[tag]
        r_max = np.sqrt(area / np.pi)
        R = r_max ** 2 / (2.0 * sag) if sag > 0 else 100.0
        rr = np.linspace(0.0, r_max, 60)
        w = rr.copy()

        for s in STANDOFFS:
            rec = next(x for x in prod if x["tag"] == tag and x["standoff_mm"] == s)
            prof = read_image(PROD / tag / f"s{s:.1f}" / "cyl_dose.mhd").flatten()
            uncp = read_image(PROD / tag / f"s{s:.1f}" / "cyl_unc.mhd").flatten()
            dep = np.array([bin_depth(i) for i in range(len(prof))])
            ok = dep >= 0.0
            dep, prof, uncp = dep[ok], prof[ok], uncp[ok]

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

        # PDD persen (s=0)
        rec0 = next(x for x in prod if x["tag"] == tag and x["standoff_mm"] == 0.0)
        prof0 = read_image(PROD / tag / "s0.0" / "cyl_dose.mhd").flatten()
        dep0 = np.array([bin_depth(i) for i in range(len(prof0))])
        m0 = (dep0 >= 0.0) & (dep0 <= 6.0)
        plt.plot(dep0[m0], 100.0 * prof0[m0] / prof0[np.argmin(abs(dep0 - 0.05))],
                 label=tag)

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
    print("TAHAP 4.2 v2 - DOSIS LAPISAN ROBUST (koreksi kurvatur)")
    print("=" * 78)
    piv = df[df.standoff_mm == 0.0].pivot_table(index="tag", columns="layer",
                                                values="dose_Gy_1GBq_60s")
    print(piv.round(3).to_string())
    print("\n--- Rasio basal/surface (kontak) ---")
    surf = df[(df.standoff_mm == 0.0) & (df.layer == "epidermis")].set_index("tag")["dose_Gy_1GBq_60s"]
    bas = df[(df.standoff_mm == 0.0) & (df.layer == "basal")].set_index("tag")["dose_Gy_1GBq_60s"]
    print((bas / surf).round(3).to_string())
    print("\n--- Efek standoff pada basal ---")
    print(df[df.layer == "basal"].pivot_table(index="tag", columns="standoff_mm",
                                              values="dose_Gy_1GBq_60s").round(3).to_string())
    print("\nTersimpan:", OUT / "robust_clinical_doses.csv", "|", OUT / "pdd_percent.png")
    print("=" * 78)


if __name__ == "__main__":
    main()