"""
Tahap 2.2 - Konversi subset tetrahedra -> STL biner per lapisan
Untuk setiap layer_*.node/.ele hasil Tahap 2.1-B v5:
1. Ekstrak muka batas (muka yang muncul tepat sekali antar tetrahedra).
2. Pertahankan orientasi outward (konvensi urutan muka tetrahedron).
3. Tulis STL biner (satuan mm, siap TessellatedVolume OpenGATE 10).
4. Uji watertight: setiap edge tak-berarah dipakai tepat 2 muka batas.
5. Cross-check: volume dari teorema divergensi permukaan vs volume tet.
Output: D:\\Brachy\\icrp156_work\\<tag>\\stl\\layer_*.stl + stl_summary.json
CLI: python 10_tet_to_stl.py [TAG ...]  (tanpa argumen = 6 set)
"""
import sys
import json
import time
from pathlib import Path
import numpy as np

WORK = Path(r"D:\Brachy\icrp156_work")
TARGET_TAGS = ["MRCP_01M", "MRCP_01F", "MRCP_05M", "MRCP_05F", "MRCP_15M", "MRCP_15F"]
LAYER_NAMES = ["epidermis", "basal", "dermis", "cranium_cortical", "cranium_spongiosa"]


def load_layer(node_p, ele_p):
    nodes = np.loadtxt(node_p, skiprows=1, usecols=(1, 2, 3), dtype=np.float64)
    ele = np.loadtxt(ele_p, skiprows=1, usecols=(1, 2, 3, 4), dtype=np.int64)
    tets = ele - 1
    return nodes, tets


def boundary_faces(tets):
    f = np.empty((len(tets), 4, 3), dtype=np.int64)
    f[:, 0] = tets[:, [0, 2, 1]]
    f[:, 1] = tets[:, [0, 1, 3]]
    f[:, 2] = tets[:, [0, 3, 2]]
    f[:, 3] = tets[:, [1, 2, 3]]
    flat = f.reshape(-1, 3)
    sorted_f = np.sort(flat, axis=1)
    uniq, inv, cnt = np.unique(sorted_f, axis=0, return_inverse=True, return_counts=True)
    mask = cnt[inv] == 1
    return flat[mask]


def watertight_check(faces):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    bad = int((cnt != 2).sum())
    return bad == 0, bad


def surface_metrics(nodes, faces):
    v0 = nodes[faces[:, 0]]
    v1 = nodes[faces[:, 1]]
    v2 = nodes[faces[:, 2]]
    vol = np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum()
    return float(vol), float(area)


def write_binary_stl(path, nodes, faces):
    v0 = nodes[faces[:, 0]].astype(np.float32)
    v1 = nodes[faces[:, 1]].astype(np.float32)
    v2 = nodes[faces[:, 2]].astype(np.float32)
    n = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(n, axis=1)
    ln[ln == 0] = 1.0
    n = (n / ln[:, None]).astype(np.float32)
    tri = np.zeros(len(faces), dtype=[("n", np.float32, (3,)), ("v", np.float32, (3, 3)), ("attr", np.uint16)])
    tri["n"] = n
    tri["v"][:, 0] = v0
    tri["v"][:, 1] = v1
    tri["v"][:, 2] = v2
    with open(path, "wb") as fh:
        fh.write(b"\0" * 80)
        fh.write(np.uint32(len(faces)).tobytes())
        fh.write(tri.tobytes())


def process_tag(tag):
    t0 = time.time()
    print(f"\n--- Konversi STL {tag} ---")
    src = WORK / tag
    dst = src / "stl"
    dst.mkdir(exist_ok=True)
    rows = []
    for name in LAYER_NAMES:
        node_p = src / f"layer_{name}.node"
        ele_p = src / f"layer_{name}.ele"
        if not node_p.exists() or not ele_p.exists():
            print(f"  [LEWATI] {name}: file subset tidak ada")
            continue
        nodes, tets = load_layer(node_p, ele_p)
        vol_tet = None
        v = nodes[tets]
        a, b, c, d = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
        vol_tet = float(np.abs(np.einsum("ij,ij->i", a - d, np.cross(b - d, c - d))).sum() / 6.0)
        faces = boundary_faces(tets)
        ok, bad_edges = watertight_check(faces)
        vol_surf, area = surface_metrics(nodes, faces)
        ratio = vol_surf / vol_tet if vol_tet > 0 else float("nan")
        write_binary_stl(dst / f"layer_{name}.stl", nodes, faces)
        rows.append({
            "tag": tag, "layer": name, "n_tri": int(len(faces)),
            "area_mm2": area, "vol_tet_mm3": vol_tet, "vol_surf_mm3": vol_surf,
            "vol_ratio": ratio, "watertight": bool(ok), "bad_edges": bad_edges,
        })
        status = "WATERTIGHT" if ok else f"BOCOR ({bad_edges} edge)"
        print(f"  {name:>18}: {len(faces):>7,} tri | luas {area:>9.1f} mm2 | "
              f"vol ratio {ratio:.4f} | {status}")
    print(f"  waktu: {time.time() - t0:.1f} detik")
    return rows


def main():
    tags = sys.argv[1:] if len(sys.argv) > 1 else TARGET_TAGS
    print("=" * 70)
    print("TAHAP 2.2 - KONVERSI TETRAHEDRA -> STL BINER PER LAPISAN")
    print(f"Set: {tags}")
    print("=" * 70)
    all_rows = []
    for tag in tags:
        all_rows.extend(process_tag(tag))
    out_json = WORK / "stl_summary.json"
    with open(out_json, "w") as f:
        json.dump(all_rows, f, indent=2)
    n_bad = sum(1 for r in all_rows if not r["watertight"])
    n_ratio_bad = sum(1 for r in all_rows if abs(r["vol_ratio"] - 1.0) > 0.01)
    print("\nRingkasan:", out_json)
    print(f"Total lapisan dikonversi : {len(all_rows)}")
    print(f"Lapisan tidak watertight : {n_bad}")
    print(f"Lapisan vol ratio >1%    : {n_ratio_bad}")
    print("=" * 70)
    print("TAHAP 2.2 SELESAI")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[KONVERSI GAGAL] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)