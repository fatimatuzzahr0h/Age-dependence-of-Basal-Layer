"""
Tahap 4.3 - Perakitan aset manuskrip (figures, tables, CSV final)
Resep dosis: sampling profil silinder pada kedalaman anatomis lokal
(unrolling kurvatur, Tahap 4.2). Uncertainty dari bin profil.
Output:
  D:\\Brachy\\stage4_output\\manuscript_layer_doses.csv
  D:\\Brachy\\stage4_output\\fig1_pdd.png ... fig4_marrow_window.png
  D:\\Brachy\\stage4_output\\manuscript_tables.md
CLI: python 28_manuscript_assets.py
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
A_BQ, T_S = 1e9, 60.0
BIN_MM, SPAN = 0.1, 23.0
TAGS = ["MRCP_01M", "MRCP_01F", "MRCP_05M", "MRCP_05F", "MRCP_15M", "MRCP_15F"]
STANDOFFS = [0.0, 0.5, 1.0]
LAYERS = ["epidermis", "basal", "dermis", "cranium_cortical", "cranium_spongiosa"]
T_BASAL = {"MRCP_01M": 0.0616, "MRCP_01F": 0.0616, "MRCP_05M": 0.0627,
           "MRCP_05F": 0.0600, "MRCP_15M": 0.0519, "MRCP_15F": 0.0511}
AGE = {t: int(t[5:7]) for t in TAGS}


def read_image(p):
    import itk
    return itk.array_from_image(itk.imread(str(p))).astype(float)


def load_profile(tag, s):
    raw = read_image(PROD / tag / f"s{s:.1f}" / "cyl_dose.mhd").flatten() / N_HIST
    unc = read_image(PROD / tag / f"s{s:.1f}" / "cyl_unc.mhd").flatten()
    dep = np.array([SPAN - BIN_MM * (i + 0.5) for i in range(len(raw))])
    ok = (dep >= 0.0) & (dep <= 22.0)
    dep, raw, unc = dep[ok], raw[ok], unc[ok]
    o = np.argsort(dep)
    return dep[o], raw[o], unc[o]


def depths_for(tag, summ):
    r = summ[tag]
    area = r["basal"]["volume_mm3"] / T_BASAL[tag]
    t = {L: r[L]["volume_mm3"] / area for L in LAYERS}
    d = {"epidermis": t["epidermis"] / 2.0,
         "basal": t["epidermis"] + t["basal"] / 2.0,
         "dermis": t["epidermis"] + t["basal"] + t["dermis"] / 2.0,
         "cranium_cortical": t["epidermis"] + t["basal"] + t["dermis"] + t["cranium_cortical"] / 2.0,
         "cranium_spongiosa": t["epidermis"] + t["basal"] + t["dermis"] + t["cranium_cortical"] + t["cranium_spongiosa"] / 2.0}
    return t, d


def main():
    OUT.mkdir(exist_ok=True)
    summ = {r["tag"]: r for r in json.load(open(WORK / "extraction_summary_v5_all.json"))}
    rows = []
    for tag in TAGS:
        t, d = depths_for(tag, summ)
        for s in STANDOFFS:
            dep, prof, unc = load_profile(tag, s)
            for L in LAYERS:
                dose = float(np.interp(d[L], dep, prof))
                u = float(np.interp(d[L], dep, unc))
                rows.append(dict(tag=tag, age_yr=AGE[tag], sex=tag[-1], standoff_mm=s,
                                 layer=L, depth_mm=round(d[L], 3),
                                 thickness_mm=round(t[L], 3),
                                 dose_Gy_per_1GBq_60s=dose * 2.0 * A_BQ * T_S,
                                 unc_rel=u))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "manuscript_layer_doses.csv", index=False)

    c0 = df[df.standoff_mm == 0.0]
    # ---- Fig 1: PDD kontak
    plt.figure(figsize=(7.2, 4.6))
    for tag in TAGS:
        dep, prof, _ = load_profile(tag, 0.0)
        m = dep <= 6.0
        plt.plot(dep[m], 100 * prof[m] / np.interp(0.05, dep, prof),
                 label=f"{AGE[tag]}y {tag[-1]}")
    plt.xlabel("Depth (mm)"); plt.ylabel("PDD (%)")
    plt.title("Fig.1  Depth-dose at vertex, contact applicator")
    plt.grid(alpha=.3); plt.legend(ncol=2, fontsize=8); plt.tight_layout()
    plt.savefig(OUT / "fig1_pdd.png", dpi=300); plt.close()

    # ---- Fig 2: dosis lapisan (kontak, log)
    plt.figure(figsize=(7.2, 4.6))
    x = np.arange(len(TAGS)); w = 0.16
    for i, L in enumerate(LAYERS):
        plt.bar(x + (i - 2) * w, c0[c0.layer == L].set_index("tag").loc[TAGS, "dose_Gy_per_1GBq_60s"],
                w, label=L)
    plt.yscale("log"); plt.xticks(x, [f"{AGE[t]}y{t[-1]}" for t in TAGS])
    plt.ylabel("Dose (Gy per GBq·60 s)"); plt.title("Fig.2  Layer doses, contact")
    plt.grid(alpha=.3, axis="y"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(OUT / "fig2_layer_doses.png", dpi=300); plt.close()

    # ---- Fig 3: retensi standoff
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    for tag in TAGS:
        for j, L in enumerate(["epidermis", "basal"]):
            sub = df[df.layer == L].set_index("standoff_mm").loc[STANDOFFS]
            sub = sub[sub.tag == tag]
            ax[j].plot(STANDOFFS, 100 * sub["dose_Gy_per_1GBq_60s"].values /
                       sub["dose_Gy_per_1GBq_60s"].values[0], "o-",
                       label=f"{AGE[tag]}y{tag[-1]}")
    ax[0].set_title("Fig.3a Epidermis retention"); ax[1].set_title("Fig.3b Basal retention")
    ax[0].set_xlabel("Standoff (mm)"); ax[1].set_xlabel("Standoff (mm)")
    ax[0].set_ylabel("% of contact dose"); ax[0].grid(alpha=.3); ax[1].grid(alpha=.3)
    ax[0].legend(fontsize=7, ncol=2); plt.tight_layout()
    plt.savefig(OUT / "fig3_standoff.png", dpi=300); plt.close()

    # ---- Fig 4: jendela sumsum
    plt.figure(figsize=(6.4, 4.4))
    ages = [1, 5, 15]
    for sex, mk in (("M", "o"), ("F", "s")):
        dep_s = [c0[(c0.tag == f"MRCP_{a:02d}{sex}") & (c0.layer == "cranium_spongiosa")]["depth_mm"].values[0] for a in ages]
        plt.plot(ages, dep_s, mk + "-", label=f"{sex} spongiosa mid-depth")
    plt.axhspan(10, 11, color="r", alpha=.15, label="beta practical range (10-11 mm)")
    plt.xlabel("Age (yr)"); plt.ylabel("Depth (mm)")
    plt.title("Fig.4  Calvarial marrow depth vs beta range")
    plt.grid(alpha=.3); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(OUT / "fig4_marrow_window.png", dpi=300); plt.close()

    # ---- Tabel markdown
    with open(OUT / "manuscript_tables.md", "w") as f:
        f.write("Table 1. Layer geometry (contact vertex region).\n\n")
        f.write(c0[c0.tag == "MRCP_01M"][["layer", "thickness_mm", "depth_mm"]].to_markdown(index=False) + "\n\n")
        f.write("Table 2. Layer doses, contact (Gy per GBq·60 s), unc in parentheses (%).\n\n")
        piv = c0.pivot_table(index="tag", columns="layer", values="dose_Gy_per_1GBq_60s")
        uncp = c0.pivot_table(index="tag", columns="layer", values="unc_rel") * 100
        tbl = piv.round(3).astype(str) + " (" + uncp.round(1).astype(str) + ")"
        f.write(tbl.to_markdown() + "\n\n")
        f.write("Table 3. Dose retention vs standoff (% of contact).\n\n")
        for L in ["epidermis", "basal"]:
            f.write(f"*{L}*\n\n")
            r = df[df.layer == L].pivot_table(index="tag", columns="standoff_mm",
                                              values="dose_Gy_per_1GBq_60s")
            f.write((100 * r.div(r[0.0], axis=0)).round(1).to_markdown() + "\n\n")

    print("=" * 78)
    print("ASET MANUSKRIP SELESAI")
    print("=" * 78)
    print(c0.pivot_table(index="tag", columns="layer",
                         values="dose_Gy_per_1GBq_60s").round(3).to_string())
    print("\nRetensi basal vs standoff (%):")
    r = df[df.layer == "basal"].pivot_table(index="tag", columns="standoff_mm",
                                            values="dose_Gy_per_1GBq_60s")
    print((100 * r.div(r[0.0], axis=0)).round(1).to_string())
    print("\nFile:", OUT / "manuscript_layer_doses.csv", "|", OUT / "manuscript_tables.md")
    print("Figur: fig1_pdd.png, fig2_layer_doses.png, fig3_standoff.png, fig4_marrow_window.png")


if __name__ == "__main__":
    main()