"""
Tahap 4.2 v4 (KOREKSI FATAL) - Dosis pada Kedalaman Anatomis Lokal
Memperbaiki jebakan bounding box pada shell melengkung.
Menggunakan ketebalan efektif Tahap 2.4 untuk "menggelar" volume menjadi
ketebalan slab lokal, lalu men-query profil silinder pada kedalaman yang benar.
TIDAK ADA SIMULASI BARU. Hanya post-processing.
Output: CORRECTED_layer_doses.csv
CLI: python 27_corrected_depths.py
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import itk

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

# Tebal basal aktual dari Tahap 2.4 (sebagai anchor untuk menghitung luas patch)
T_BASAL_ANCHOR = {"MRCP_01M": 0.0616, "MRCP_01F": 0.0616, "MRCP_05M": 0.0627,
                  "MRCP_05F": 0.0600, "MRCP_15M": 0.0519, "MRCP_15F": 0.0511}

def read_image(p):
    return itk.array_from_image(itk.imread(str(p))).astype(float)

def bin_depth(i):
    return SPAN_DEPTH - BIN_MM * (i + 0.5)

def load_profile(tag, s):
    raw = read_image(PROD / tag / f"s{s:.1f}" / "cyl_dose.mhd").flatten() / N_HIST
    dep = np.array([bin_depth(i) for i in range(len(raw))])
    ok = (dep >= 0.0) & (dep <= 22.0)
    dep, raw = dep[ok], raw[ok]
    order = np.argsort(dep)
    return dep[order], raw[order]

def main():
    OUT.mkdir(exist_ok=True)
    summ = {r["tag"]: r for r in json.load(open(WORK / "extraction_summary_v5_all.json"))}
    rows = []

    for tag in TAGS:
        r = summ[tag]
        # Hitung luas patch lokal menggunakan volume basal dan tebal anchor-nya
        area = r["basal"]["volume_mm3"] / T_BASAL_ANCHOR[tag]
        
        # Hitung tebal lokal setiap lapisan (Volume / Luas)
        t = {L: r[L]["volume_mm3"] / area for L in LAYERS}
        
        # Hitung kedalaman titik tengah (midpoint) setiap lapisan secara berurutan
        d = {}
        d["epidermis"] = t["epidermis"] / 2.0
        d["basal"] = t["epidermis"] + t["basal"] / 2.0
        d["dermis"] = d["basal"] + t["basal"]/2.0 + t["dermis"] / 2.0
        d["cranium_cortical"] = d["dermis"] + t["dermis"]/2.0 + t["cranium_cortical"] / 2.0
        d["cranium_spongiosa"] = d["cranium_cortical"] + t["cranium_cortical"]/2.0 + t["cranium_spongiosa"] / 2.0

        print(f"\n--- {tag} | Kedalaman Anatomis Lokal (mm) ---")
        for L in LAYERS:
            print(f"  {L:18}: {d[L]:.3f} mm")

        for s in STANDOFFS:
            dep, prof = load_profile(tag, s)
            for L in LAYERS:
                # Query profil pada kedalaman yang BENAR
                dose_hist = float(np.interp(d[L], dep, prof))
                dose_clin = dose_hist * 2.0 * ACTIVITY_BQ * TIME_S
                rows.append(dict(tag=tag, standoff_mm=s, layer=L,
                                 depth_mm=round(d[L], 3),
                                 dose_Gy_1GBq_60s=dose_clin))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "CORRECTED_layer_doses.csv", index=False)

    print("\n" + "="*78)
    print("DOSIS LAPISAN TERKOREKSI ( Gy per 1 GBq x 60 s )")
    print("="*78)
    piv = df[df.standoff_mm == 0.0].pivot_table(index="tag", columns="layer", values="dose_Gy_1GBq_60s")
    piv = piv[LAYERS] # Urutkan kolom
    print(piv.round(3).to_string())

    print("\n--- Rasio basal/epidermis (kontak) ---")
    e = df[(df.standoff_mm == 0.0) & (df.layer == "epidermis")].set_index("tag")["dose_Gy_1GBq_60s"]
    b = df[(df.standoff_mm == 0.0) & (df.layer == "basal")].set_index("tag")["dose_Gy_1GBq_60s"]
    print((b / e).round(3).to_string())
    print("^(Kini harus < 1.0, mencerminkan attenuasi beta)")

if __name__ == "__main__":
    main()