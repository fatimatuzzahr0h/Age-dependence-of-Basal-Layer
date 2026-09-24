"""
Tahap 2.1-C - Diagnostik ground-truth mesh ICRP-156 (read-only, cepat)
Menjawab 3 pertanyaan fondasional sebelum ekstraksi boleh dipercaya:
[1] Satuan koordinat .node (mm atau cm) via span-z vs tinggi badan referensi.
[2] Basis indeks .ele (0-based atau 1-based) via min/max indeks sudut.
[3] Volume tubuh total dari mesh vs massa tubuh referensi (cross-check satuan).
[4] Rentang z centroid untuk ID 12200/12201/2600 dengan indeks yang benar.
Cara jalankan (cepat, ~1 menit/set):
python 09_mesh_ground_truth.py MRCP_01M MRCP_15F
"""
import sys
from pathlib import Path
import numpy as np

RAW = Path(r"D:\Brachy\icrp156_raw")

# Referensi pendekatan (ICRP-89/156): tinggi (cm) dan massa (kg)
REF = {
    "MRCP_01M": (74.5, 10.0), "MRCP_01F": (74.5, 10.0),
    "MRCP_05M": (109.5, 19.0), "MRCP_05F": (109.5, 19.0),
    "MRCP_15M": (170.0, 56.0), "MRCP_15F": (158.5, 50.0),
}


def main():
    tags = sys.argv[1:] if len(sys.argv) > 1 else ["MRCP_01M", "MRCP_15F"]
    for tag in tags:
        hits = sorted(RAW.rglob(f"{tag}.node"))
        if not hits:
            print(f"{tag}: tidak ditemukan")
            continue
        node_p = hits[0]
        ele_p = node_p.with_suffix(".ele")
        print("=" * 70)
        print(f"DIAGNOSTIK GROUND-TRUTH: {tag}")
        print("=" * 70)

        nodes = np.loadtxt(node_p, skiprows=1, usecols=(1, 2, 3), dtype=np.float64)
        n_nodes = len(nodes)
        lo = nodes.min(axis=0)
        hi = nodes.max(axis=0)
        span = hi - lo
        st_cm, mass_kg = REF.get(tag, (float("nan"), float("nan")))
        ratio = span[2] / st_cm
        unit = "cm" if 0.9 <= ratio <= 1.1 else ("mm" if 9.0 <= ratio <= 11.0 else "TIDAK DIKETAHUI")
        print(f"[1] NODE: N = {n_nodes:,}")
        print(f"    min = ({lo[0]:.2f}, {lo[1]:.2f}, {lo[2]:.2f})")
        print(f"    max = ({hi[0]:.2f}, {hi[1]:.2f}, {hi[2]:.2f})")
        print(f"    span-z = {span[2]:.2f} vs tinggi referensi {st_cm:.1f} cm -> rasio {ratio:.2f}")
        print(f"    => KESIMPULAN SATUAN: {unit}")

        ele_raw = np.loadtxt(ele_p, skiprows=1, usecols=(1, 2, 3, 4), dtype=np.int64)
        cmin = int(ele_raw.min())
        cmax = int(ele_raw.max())
        print(f"[2] ELE: indeks sudut min = {cmin}, max = {cmax}, N_node = {n_nodes}")
        if cmin == 0 and cmax <= n_nodes - 1:
            base = 0
            print("    => BASIS 0: skrip v1-v4 SALAH karena mengurangkan 1!")
        elif cmin == 1 and cmax <= n_nodes:
            base = 1
            print("    => BASIS 1: pengurangan 1 di v1-v4 sudah benar.")
        else:
            base = 1
            print("    => AMBIGU: diasumsikan basis 1, periksa manual!")

        ids = np.loadtxt(ele_p, skiprows=1, usecols=(5,), dtype=np.int64)
        tets = ele_raw - base

        vol = 0.0
        CH = 500000
        for i in range(0, len(tets), CH):
            v = nodes[tets[i:i + CH]]
            a, b, c, d = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
            vol += np.abs(np.einsum("ij,ij->i", a - d, np.cross(b - d, c - d))).sum() / 6.0
        print(f"[3] Volume mesh total = {vol:.6e} (satuan berkas)^3")
        print(f"    jika satuan cm -> {vol/1000.0:.2f} kg (referensi {mass_kg:.1f} kg)")
        print(f"    jika satuan mm -> {vol/1000.0/1000.0:.2f} kg (referensi {mass_kg:.1f} kg)")

        print("[4] Rentang z centroid (indeks benar):")
        for oid in (12200, 12201, 2600):
            m = ids == oid
            tt = tets[m]
            cent = nodes[tt].mean(axis=1)
            print(f"    ID {oid}: n_tet = {m.sum():,} | z centroid [{cent[:,2].min():.2f}, {cent[:,2].max():.2f}]")
        print("")


if __name__ == "__main__":
    main()