"""
Tahap 2.0 - Inspeksi read-only berkas mesh ICRP-156 (v3: rekursif + filter)
Keputusan metodologis terkunci: matriks produksi = 18 skenario
(1, 5, 15 tahun x L/P x 3 jarak). Set neonatus (00M/00F) DILEWATI.
Skrip mencari .node/.ele secara rekursif di subfolder, hanya untuk
6 set target, lalu melaporkan:
1. Header .node dan .ele (format TetGen)
2. Jumlah tetrahedra per Organ ID (kolom atribut terakhir di .ele)
3. Nama material/organ dari .mtl (kandidat pemetaan ID -> nama)
4. Pratinjau berkas .dat (komposisi media)
Skrip ini TIDAK mengubah berkas apa pun (read-only).
Cara jalankan:
python 06_icrp156_inspection.py
"""
import sys
import json
from pathlib import Path
import numpy as np

RAW_DIR = Path(r"D:\Brachy\icrp156_raw")
OUT_JSON = Path(r"D:\Brachy\icrp156_inspection.json")

TARGET_TAGS = ["MRCP_01M", "MRCP_01F", "MRCP_05M", "MRCP_05F", "MRCP_15M", "MRCP_15F"]


def read_first_two_lines(path):
    with open(path, "r", errors="ignore") as f:
        header = f.readline()
        first = f.readline()
    return header.split(), first.split()


def inspect_set(node_p, ele_p, mtl_p, tag):
    info = {"tag": tag, "folder": str(node_p.parent)}
    if not node_p.exists() or not ele_p.exists():
        info["error"] = "file .node atau .ele tidak ditemukan"
        return info

    nh, nf = read_first_two_lines(node_p)
    info["node_header"] = dict(
        count=int(nh[0]), dim=int(nh[1]),
        attrs=int(nh[2]) if len(nh) > 2 else 0,
        markers=int(nh[3]) if len(nh) > 3 else 0)

    eh, ef = read_first_two_lines(ele_p)
    n_tets = int(eh[0])
    corners = int(eh[1])
    attrs = int(eh[2]) if len(eh) > 2 else 0
    ncols = len(ef)
    info["ele_header"] = dict(count=n_tets, corners=corners, attrs=attrs)
    info["ele_first_row_ncols"] = ncols

    usecol = ncols - 1
    ids = np.loadtxt(ele_p, skiprows=1, usecols=(usecol,), dtype=np.int64)
    uniq, cnt = np.unique(ids, return_counts=True)
    order = np.argsort(-cnt)
    info["organ_ids"] = [{"id": int(uniq[i]), "n_tets": int(cnt[i])} for i in order]

    if mtl_p.exists():
        names = []
        with open(mtl_p, "r", errors="ignore") as f:
            for line in f:
                if line.startswith("newmtl"):
                    names.append(line.split()[1])
        info["mtl_names"] = names

    parent = node_p.parent
    dats = sorted(parent.glob("*.dat"))
    prev = {}
    for d in dats:
        try:
            with open(d, "r", errors="ignore") as f:
                prev[d.name] = [f.readline().rstrip() for _ in range(5)]
        except Exception:
            pass
    info["dat_preview"] = prev
    return info


def main():
    print("=" * 70)
    print("TAHAP 2.0 - INSPEKSI READ-ONLY ICRP-156 (6 SET TARGET)")
    print("=" * 70)
    if not RAW_DIR.exists():
        print(f"ERROR: folder tidak ditemukan: {RAW_DIR}")
        sys.exit(1)

    all_nodes = sorted(RAW_DIR.rglob("MRCP_*.node"))
    all_tags = sorted({p.stem for p in all_nodes})
    skipped = [t for t in all_tags if t not in TARGET_TAGS]
    print(f"Set ditemukan di folder : {all_tags}")
    print(f"Set target inspeksi     : {TARGET_TAGS}")
    print(f"Set dilewati (di luar scope): {skipped}")

    results = []
    for tag in TARGET_TAGS:
        hits = [p for p in all_nodes if p.stem == tag]
        if not hits:
            print(f"\n--- {tag}: TIDAK DITEMUKAN ---")
            results.append({"tag": tag, "error": "tidak ditemukan"})
            continue
        node_p = hits[0]
        ele_p = node_p.with_suffix(".ele")
        mtl_p = node_p.with_suffix(".mtl")
        print(f"\n--- Inspeksi {tag} (subfolder: {node_p.parent.name}) ---")
        info = inspect_set(node_p, ele_p, mtl_p, tag)
        results.append(info)
        if "error" in info:
            print("  ", info["error"])
            continue
        nh = info["node_header"]
        eh = info["ele_header"]
        print(f"  Node : {nh['count']:,} (dim {nh['dim']}, attrs {nh['attrs']}, markers {nh['markers']})")
        print(f"  Tet  : {eh['count']:,} (corners {eh['corners']}, attrs {eh['attrs']}, kolom/baris {info['ele_first_row_ncols']})")
        print(f"  Organ ID unik: {len(info['organ_ids'])}")
        for row in info["organ_ids"][:15]:
            print(f"    ID {row['id']:>5} : {row['n_tets']:>12,} tetrahedra")
        if len(info["organ_ids"]) > 15:
            print(f"    ... dan {len(info['organ_ids']) - 15} ID lain")
        if "mtl_names" in info:
            print(f"  Nama di .mtl ({len(info['mtl_names'])}): {info['mtl_names'][:20]}")
            if len(info["mtl_names"]) > 20:
                print(f"    ... dan {len(info['mtl_names']) - 20} nama lain")
        if info["dat_preview"]:
            for name, lines in list(info["dat_preview"].items())[:3]:
                print(f"  Pratinjau {name}:")
                for ln in lines:
                    print(f"    | {ln}")

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print("\nRingkasan JSON disimpan di:", OUT_JSON)
    print("=" * 70)
    print("TAHAP 2.0 (INSPEKSI) SELESAI")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[INSPEKSI GAGAL] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)