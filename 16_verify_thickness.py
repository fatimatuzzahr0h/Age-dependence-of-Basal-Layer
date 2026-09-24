"""
Tahap 2.4 - Verifikasi Ketebalan Efektif vs Referensi ICRP
Menghitung ketebalan efektif lapisan = Volume / Luas Proyeksi Silinder.
Membandingkan dengan referensi histologi ICRP 89/156.
Cara jalankan:
python 16_verify_thickness.py
"""
import json
import numpy as np
from pathlib import Path

STAGE23_JSON = Path(r"D:\Brachy\stage23_output\stage23_summary.json")
# Luas proyeksi silinder verteks (r = 20 mm)
AREA_MM2 = np.pi * (20.0 ** 2) 

# Referensi ketebalan ICRP (pendekatan untuk kulit kepala)
REF_ICRP = {
    "epidermis": "50 - 100 µm (bervariasi theo usia, lebih tebal di remaja)",
    "basal": "40 - 50 µm (lapisan target radiobiologis)",
    "cranium_cortical": "1 - 5 mm (menebal signifikan dari balita ke remaja)",
    "cranium_spongiosa": "2 - 8 mm (diploë, tempat sumsum tulang)"
}

def main():
    with open(STAGE23_JSON, "r") as f:
        data = json.load(f)
        
    print("=" * 80)
    print("TAHAP 2.4 - VERIFIKASI KETEBALAN EFEKTIF vs REFERENSI ICRP")
    print("=" * 80)
    print(f"{'Set':<10} | {'Lapisan':<20} | {'Vol (mm3)':>10} | {'Tebal Efektif':>14} | Referensi ICRP")
    print("-" * 80)
    
    for row in data:
        tag = row["tag"]
        layer = row["layer"]
        vol = row["volume_mm3"]
        
        # Ketebalan efektif dalam mm, lalu konversi ke mikrometer (µm) untuk kulit
        thick_mm = vol / AREA_MM2
        if layer in ["epidermis", "basal"]:
            thick_str = f"{thick_mm * 1000:6.1f} µm"
        else:
            thick_str = f"{thick_mm:6.2f} mm"
            
        ref = REF_ICRP.get(layer, "-")
        print(f"{tag:<10} | {layer:<20} | {vol:10.1f} | {thick_str:>14} | {ref}")
        
    print("=" * 80)
    print("INSTRUKSI VISUALISASI (QA MORFOLOGI):")
    print("1. Buka File Explorer -> D:\\Brachy\\icrp156_work\\MRCP_01M\\stl\\")
    print("2. Klik kanan file 'layer_basal.stl' -> Open with -> 3D Viewer (bawaan Windows)")
    print("   (Atau gunakan MeshLab/Blender jika terinstal).")
    print("3. Pastikan bentuknya berupa kubah/cangkang mulus tanpa lubang (watertight).")
    print("=" * 80)

if __name__ == "__main__":
    main()