"""
Tahap 5 (rencana asli) - ANALISIS DATA FORMAL 5.1-5.4
Menutup dua gap Tahap 4 dan memformalkan output analisis:
 5.1 tabel rasio basal/epidermis 18 skenario + propagasi uncertainty MC
 5.2 deskriptif per usia & per gender (kontak) dengan interval MC;
     TANPA uji hipotesis (n kecil per sel; sesuai rencana)
 5.3 sensitivitas standoff: retensi + kehilangan per mm
 5.4 evaluasi sumsum: rasio spongiosa/epidermis vs kedalaman vs jangkauan beta
Input : D:\\Brachy\\stage4_output\\manuscript_layer_doses.csv
Output: D:\\Brachy\\stage5_output\\stage5_ratios.csv
        D:\\Brachy\\stage5_output\\stage5_analysis.md
CLI : python 29_stage5_analysis.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

SRC = Path(r"D:\Brachy\stage4_output\manuscript_layer_doses.csv")
OUT = Path(r"D:\Brachy\stage5_output")
LAYERS = ["epidermis", "basal", "dermis", "cranium_cortical", "cranium_spongiosa"]
RANGE_BETA = (10.0, 11.0)  # practical range mm


def df_to_md(df, index=False):
    cols = [str(c) for c in df.columns]
    rows = [[str(v) for v in r] for r in df.values]
    if index:
        cols = [""] + cols
        rows = [[str(i)] + r for i, r in zip(df.index, rows)]
    w = [max(len(c), max((len(r[i]) for r in rows), default=0)) for i, c in enumerate(cols)]
    hdr = "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(cols)) + " |"
    sep = "|-" + "-|-".join("-" * x for x in w) + "-|"
    body = "\n".join("| " + " | ".join(r[i].ljust(w[i]) for i in range(len(cols))) + " |" for r in rows)
    return hdr + "\n" + sep + "\n" + body


def get(df, tag, s, layer):
    r = df[(df.tag == tag) & (df.standoff_mm == s) & (df.layer == layer)]
    return r["dose_Gy_per_1GBq_60s"].values[0], r["unc_rel"].values[0]


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(SRC)
    tags = list(dict.fromkeys(df["tag"]))
    stands = sorted(df["standoff_mm"].unique())
    lines = []

    # ---------- 5.1 ----------
    rows = []
    for t in tags:
        age = int(df[df.tag == t]["age_yr"].values[0])
        sex = df[df.tag == t]["sex"].values[0]
        for s in stands:
            a, ua = get(df, t, s, "basal")
            b, ub = get(df, t, s, "epidermis")
            rows.append(dict(tag=t, age_yr=age, sex=sex, standoff_mm=s,
                             basal_over_epidermis=round(a / b, 3),
                             unc_rel=round(float(np.sqrt(ua**2 + ub**2)), 4)))
    r51 = pd.DataFrame(rows)
    r51.to_csv(OUT / "stage5_ratios.csv", index=False)
    lines += ["## 5.1 Rasio basal/epidermis (18 skenario)", "", df_to_md(r51), ""]

    # ---------- 5.2 ----------
    c0 = df[df.standoff_mm == 0.0]
    rows = []
    for L in LAYERS:
        for age in [1, 5, 15]:
            vM, uM = get(c0, f"MRCP_{age:02d}M", 0.0, L)
            vF, uF = get(c0, f"MRCP_{age:02d}F", 0.0, L)
            m = 0.5 * (vM + vF)
            um = 0.5 * np.sqrt((vM * uM) ** 2 + (vF * uF) ** 2)
            rows.append(dict(layer=L, age_yr=age, M=round(vM, 3), F=round(vF, 3),
                             mean=round(m, 3), mc_unc_pct=round(100 * um / m, 2),
                             diff_MF_pct=round(100 * (vM - vF) / vF, 1)))
    r52 = pd.DataFrame(rows)
    lines += ["## 5.2 Deskriptif per usia & gender (kontak, Gy per GBq.60s)",
              "Catatan: interval = uncertainty Monte Carlo saja (statistik);",
              "variabilitas antar-individu tidak disampling (limitasi).", "", df_to_md(r52), ""]

    # ---------- 5.3 ----------
    rows = []
    for t in tags:
        d = {}
        for L in ["epidermis", "basal"]:
            a0, _ = get(df, t, 0.0, L)
            a05, _ = get(df, t, 0.5, L)
            a10, _ = get(df, t, 1.0, L)
            d[f"{L}_ret0.5"] = round(100 * a05 / a0, 1)
            d[f"{L}_ret1.0"] = round(100 * a10 / a0, 1)
            d[f"{L}_loss_per_mm"] = round(100 * (1 - a10 / a0), 1)
        rows.append(dict(tag=t, **d))
    r53 = pd.DataFrame(rows)
    lines += ["## 5.3 Sensitivitas standoff (% retensi & kehilangan per mm)", "", df_to_md(r53), ""]

    # ---------- 5.4 ----------
    rows = []
    for t in tags:
        age = int(df[df.tag == t]["age_yr"].values[0])
        sex = df[df.tag == t]["sex"].values[0]
        sE, uE = get(c0, t, 0.0, "epidermis")
        sS, uS = get(c0, t, 0.0, "cranium_spongiosa")
        dep = c0[(c0.tag == t) & (c0.layer == "cranium_spongiosa")]["depth_mm"].values[0]
        zone = ("DALAM jangkauan beta" if dep < RANGE_BETA[0] else
                "TEPI jangkauan beta" if dep <= RANGE_BETA[1] else "LUAR jangkauan beta")
        rows.append(dict(tag=t, age_yr=age, sex=sex, spong_depth_mm=dep,
                         spong_over_epi_pct=round(100 * sS / sE, 1),
                         unc_pct=round(100 * float(np.sqrt(uS**2 + uE**2)), 1),
                         zone=zone))
    r54 = pd.DataFrame(rows)
    lines += ["## 5.4 Sumsum kalvaria: rasio dosis & zona jangkauan beta", "", df_to_md(r54), ""]

    (OUT / "stage5_analysis.md").write_text("\n".join(lines), encoding="utf-8")

    print("=" * 78)
    print("TAHAP 5 (ASLI) - ANALISIS DATA FORMAL")
    print("=" * 78)
    print("\n[5.1] Rasio basal/epidermis, kontak:")
    print(r51[r51.standoff_mm == 0.0].to_string(index=False))
    print("\n[5.2] Ringkasan usia (basal, kontak):")
    print(r52[r52.layer == "basal"].to_string(index=False))
    print("\n[5.3] Retensi basal vs standoff:")
    print(r53[["tag", "basal_ret0.5", "basal_ret1.0", "basal_loss_per_mm"]].to_string(index=False))
    print("\n[5.4] Zona sumsum:")
    print(r54.to_string(index=False))
    print("\nTersimpan:", OUT / "stage5_ratios.csv", "|", OUT / "stage5_analysis.md")
    print("=" * 78)


if __name__ == "__main__":
    main()