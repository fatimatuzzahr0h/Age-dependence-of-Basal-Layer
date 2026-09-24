"""
Tahap 2.1-A - Pemetaan Organ ID ke nama organ (read-only)
Menggunakan tiga sumber lokal:
1. .mtl  : nama material berformat <ID>_<nama>
2. bone.dat : baris situs tulang (termasuk Cranium) + rasio RBM/YBM/TB/CB/MST
3. media.dat: baris komposisi medium (termasuk skin/spongiosa bila ada)
Sekalian menerjemahkan Organ ID peringkat teratas dari icrp156_inspection.json
agar kita tahu organ apa yang paling dominan di setiap set.
Skrip ini TIDAK mengubah berkas apa pun.
Cara jalankan:
python 07_icrp156_mapping.py
"""
import sys
import json
import re
from pathlib import Path

RAW_DIR = Path(r"D:\Brachy\icrp156_raw")
INSPECT_JSON = Path(r"D:\Brachy\icrp156_inspection.json")
OUT_JSON = Path(r"D:\Brachy\icrp156_mapping.json")

TARGET_TAGS = ["MRCP_01M", "MRCP_01F", "MRCP_05M", "MRCP_05F", "MRCP_15M", "MRCP_15F"]

KEYWORDS = ["skin", "cranium", "skull", "marrow", "spongiosa", "cortical",
            "head", "scalp", "epider", "dermis", "basal", "brain", "meninges"]


def find_set_folder(tag):
    hits = sorted(RAW_DIR.rglob(f"{tag}.node"))
    if not hits:
        return None
    return hits[0].parent


def parse_mtl(path):
    mapping = {}
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("newmtl"):
                parts = line.split()
                if len(parts) < 2:
                    continue
                name = parts[1]
                m = re.match(r"^(\d+)_(.+)$", name)
                if m:
                    mapping[int(m.group(1))] = m.group(2)
    return mapping


def grep_lines(path, keywords):
    hits = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            low = line.lower()
            if any(k in low for k in keywords):
                hits.append(line.rstrip())
    return hits


def main():
    print("=" * 70)
    print("TAHAP 2.1-A - PEMETAAN ORGAN ID -> NAMA (READ-ONLY)")
    print("=" * 70)

    inspect = {}
    if INSPECT_JSON.exists():
        with open(INSPECT_JSON, "r") as f:
            for row in json.load(f):
                inspect[row.get("tag")] = row

    allmap = {}
    for tag in TARGET_TAGS:
        folder = find_set_folder(tag)
        if folder is None:
            print(f"\n--- {tag}: folder tidak ditemukan ---")
            continue
        mtl_p = folder / f"{tag}.mtl"
        if not mtl_p.exists():
            print(f"\n--- {tag}: file .mtl tidak ditemukan ---")
            continue
        mapping = parse_mtl(mtl_p)
        allmap[tag] = {str(k): v for k, v in mapping.items()}

        print(f"\n--- {tag}: {len(mapping)} nama organ terbaca dari .mtl ---")

        print("  [A] Nama organ terkait kulit kepala / tulang / sumsum:")
        n_hit = 0
        for oid in sorted(mapping):
            low = mapping[oid].lower()
            if any(k in low for k in KEYWORDS):
                print(f"      ID {oid:>6} : {mapping[oid]}")
                n_hit += 1
        if n_hit == 0:
            print("      (tidak ada yang cocok kata kunci)")

        info = inspect.get(tag)
        if info and "organ_ids" in info:
            print("  [B] Terjemahan Organ ID peringkat teratas (dari inspeksi):")
            for row in info["organ_ids"][:12]:
                oid = row["id"]
                nm = mapping.get(oid, "??? tidak ada di .mtl")
                print(f"      ID {oid:>6} : {row['n_tets']:>10,} tet : {nm}")

        bone_p = folder / f"{tag}_bone.dat"
        if bone_p.exists():
            hits = grep_lines(bone_p, ["cranium", "skull", "head"])
            print(f"  [C] Baris Cranium/Skull di bone.dat ({len(hits)} baris):")
            for h in hits[:12]:
                print(f"      | {h}")

        media_p = folder / f"{tag}_media.dat"
        if media_p.exists():
            hits = grep_lines(media_p, ["skin", "spongiosa", "cranium", "marrow"])
            print(f"  [D] Baris skin/spongiosa/marrow di media.dat ({len(hits)} baris):")
            for h in hits[:12]:
                print(f"      | {h}")

    with open(OUT_JSON, "w") as f:
        json.dump(allmap, f, indent=2)
    print("\nPemetaan lengkap disimpan di:", OUT_JSON)
    print("=" * 70)
    print("TAHAP 2.1-A (PEMETAAN) SELESAI")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[PEMETAAN GAGAL] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)