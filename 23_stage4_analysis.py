"""
Tahap 4.1 - Analisis Kurva PDD & Tabel Dosis Klinis
Menggunakan data Aktor Silinder (Profile Cylinder) sebagai ground truth
karena Aktor STL mengalami artifact navigasi G4TessellatedSolid pada
lapisan tipis (<100 um).
Menghitung dosis klinis untuk protokol referensi: 1 GBq, 60 detik.
Output: 
- D:\\Brachy\\stage4_output\\pdd_curves.png
- D:\\Brachy\\stage4_output\\clinical_doses.csv
CLI: python 23_stage4_analysis.py
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(r"D:\Brachy\stage4_output")
PROD_JSON = Path(r"D:\Brachy\production_output\production_summary.json")

# Protokol Klinis Referensi
ACTIVITY_BQ = 1e9  # 1 GBq
TIME_S = 60        # 60 detik
TOTAL_BQ_S = ACTIVITY_BQ * TIME_S

def main():
    OUT.mkdir(exist_ok=True)
    with open(PROD_JSON, "r") as f:
        data = json.load(f)
        
    print("=" * 78)
    print("TAHAP 4.1 - ANALISIS PDD & DOSIS KLINIS (Ground Truth: Silinder)")
    print("=" * 78)
    
    # 1. Plot Kurva PDD (Standoff = 0.0 mm)
    plt.figure(figsize=(10, 6))
    depths_mm = [0.05, 0.1, 0.5, 0.7, 1.0, 2.0, 3.0, 5.0]
    
    clinical_rows = []
    
    # Group by tag
    tags = sorted(list(set([r["tag"] for r in data])))
    
    for tag in tags:
        # Ambil data standoff 0.0
        row_s0 = next(r for r in data if r["tag"] == tag and r["standoff_mm"] == 0.0)
        
        profile = row_s0["profile"]
        doses_gy = [profile[f"{d}mm"]["dose_Gy_per_Bq_s"] * TOTAL_BQ_S for d in depths_mm]
        
        plt.plot(depths_mm, doses_gy, marker='o', label=f"{tag} (s=0.0)")
        
        # Ekstrak dosis klinis untuk tabel
        # Epidermis proxy = 0.05 mm
        # Basal proxy = 0.7 mm (kedalaman efektif basal)
        # Dermis proxy = 2.0 mm
        epi_dose = profile["0.05mm"]["dose_Gy_per_Bq_s"] * TOTAL_BQ_S
        basal_dose = profile["0.7mm"]["dose_Gy_per_Bq_s"] * TOTAL_BQ_S
        dermis_dose = profile["2.0mm"]["dose_Gy_per_Bq_s"] * TOTAL_BQ_S
        
        clinical_rows.append({
            "Phantom": tag,
            "Epidermis (Gy)": epi_dose,
            "Basal (Gy)": basal_dose,
            "Dermis (Gy)": dermis_dose,
            "Epidermis Unc (%)": profile["0.05mm"]["unc_rel"] * 100,
            "Basal Unc (%)": profile["0.7mm"]["unc_rel"] * 100
        })
        
    plt.title("Percentage Depth Dose (PDD) - Sr-90/Y-90 Applicator (1 GBq, 60s, Contact)", fontsize=14)
    plt.xlabel("Kedalaman Jaringan (mm)", fontsize=12)
    plt.ylabel("Dosis Absolut (Gy)", fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend(title="Phantom Set")
    plt.tight_layout()
    plt.savefig(OUT / "pdd_curves.png", dpi=300)
    print(f"[OK] Grafik PDD disimpan: {OUT / 'pdd_curves.png'}")
    
    # 2. Buat Tabel Dosis Klinis
    df_clin = pd.DataFrame(clinical_rows)
    df_clin.to_csv(OUT / "clinical_doses.csv", index=False)
    print(f"[OK] Tabel dosis klinis disimpan: {OUT / 'clinical_doses.csv'}")
    
    print("\n--- Ringkasan Dosis Klinis (1 GBq, 60 detik) ---")
    print(df_clin.to_string(index=False))
    
    # 3. Analisis Pengaruh Standoff (Contoh pada 01M)
    print("\n--- Pengaruh Standoff pada MRCP_01M (Dosis Basal / 0.7mm) ---")
    for s in [0.0, 0.5, 1.0]:
        row = next(r for r in data if r["tag"] == "MRCP_01M" and r["standoff_mm"] == s)
        d_basal = row["profile"]["0.7mm"]["dose_Gy_per_Bq_s"] * TOTAL_BQ_S
        print(f"  Standoff {s:.1f} mm: {d_basal:.3f} Gy")
        
    print("\n" + "=" * 78)
    print("TAHAP 4.1 SELESAI")
    print("Catatan Manuskrip: Dosis diturunkan dari profil silinder voxel (bebas")
    print("artifact navigasi G4TessellatedSolid). Limitasi L7 (finite patch size)")
    print("berlaku untuk dosis absolut permukaan.")
    print("=" * 78)

if __name__ == "__main__":
    main()