"""
Tahap 2.3-fix - Re-centering STL ke origin verteks (koordinat terkunci di file)
Alasan: TesselatedVolume melakukan re-basing saat load; translasi runtime
(-x0,-y0,-z_top) menyebabkan pergeseran ganda (solid terlempar ke z ~ -381 mm,
di luar world +/-200 mm). Maka centering dibaked ke dalam file:
  koordinat_baru = koordinat_absolut - (x0, y0, z_top)
sehingga verteks persis z=0 dan pusat xy=(0,0); translasi runtime dinolkan.
Output: layer_*.node tertimpa (ter-center) + layer_*.stl ditulis ulang.
Read-only terhadap icrp156_raw.
CLI: python 15_recenter_stl.py
"""
import json
import numpy as np
from pathlib import Path

WORK = Path(r"D:\Brachy\icrp156_work")
LAYERS = ["epidermis", "basal", "dermis", "cranium_cortical", "cranium_spongiosa"]


def boundary_faces(tets):
    f = np.empty((len(tets), 4, 3), dtype=np.int64)
    f[:, 0] = tets[:, [0, 2, 1]]
    f[:, 1] = tets[:, [0, 1, 3]]
    f[:, 2] = tets[:, [0, 3, 2]]
    f[:, 3] = tets[:, [1, 2, 3]]
    flat = f.reshape(-1, 3)
    srt = np.sort(flat, axis=1)
    uniq, inv, cnt = np.unique(srt, axis=0, return_inverse=True, return_counts=True)
    return flat[cnt[inv] == 1]


def watertight(faces):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    return int((cnt != 2).sum())


def write_binary_stl(path, nodes, faces):
    v0 = nodes[faces[:, 0]].astype(np.float32)
    v1 = nodes[faces[:, 1]].astype(np.float32)
    v2 = nodes[faces[:, 2]].astype(np.float32)
    n = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(n, axis=1)
    ln[ln == 0] = 1.0
    tri = np.zeros(len(faces), dtype=[("n", np.float32, (3,)), ("v", np.float32, (3, 3)), ("attr", np.uint16)])
    tri["n"] = (n / ln[:, None]).astype(np.float32)
    tri["v"][:, 0] = v0
    tri["v"][:, 1] = v1
    tri["v"][:, 2] = v2
    with open(path, "wb") as fh:
        fh.write(b"\0" * 80)
        fh.write(np.uint32(len(faces)).tobytes())
        fh.write(tri.tobytes())


def main():
    with open(WORK / "extraction_summary_v5_all.json", "r") as f:
        summ = json.load(f)
    print("=" * 70)
    print("RE-CENTERING STL KE ORIGIN VERTEKS (6 set x 5 lapisan)")
    print("=" * 70)
    for row in summ:
        tag = row["tag"]
        x0, y0 = row["xy_center_mm"]
        z0 = row["z_top_mm"]
        off = np.array([x0, y0, z0], dtype=np.float64)
        src = WORK / tag
        dst = src / "stl"
        print(f"\n--- {tag}: offset ({x0:.2f}, {y0:.2f}, {z0:.2f}) mm ---")
        for name in LAYERS:
            node_p = src / f"layer_{name}.node"
            ele_p = src / f"layer_{name}.ele"
            if not node_p.exists() or not ele_p.exists():
                print(f"  [LEWATI] {name}")
                continue
            nodes = np.loadtxt(node_p, skiprows=1, usecols=(1, 2, 3), dtype=np.float64)
            ele = np.loadtxt(ele_p, skiprows=1, usecols=(1, 2, 3, 4), dtype=np.int64)
            tets = ele - 1
            nodes_c = nodes - off
            np.savetxt(node_p,
                       np.column_stack([np.arange(1, len(nodes_c) + 1), nodes_c]),
                       fmt=["%d", "%.4f", "%.4f", "%.4f"],
                       header=f"{len(nodes_c)} 3 0 0", comments="")
            faces = boundary_faces(tets)
            bad = watertight(faces)
            write_binary_stl(dst / f"layer_{name}.stl", nodes_c, faces)
            zmin, zmax = nodes_c[:, 2].min(), nodes_c[:, 2].max()
            print(f"  {name:>18}: {len(faces):>7,} tri | z [{zmin:8.2f},{zmax:7.2f}] mm | "
                  f"bad edges {bad}")
    print("\n" + "=" * 70)
    print("RE-CENTERING SELESAI | verteks kini z=0, pusat xy=(0,0)")
    print("=" * 70)


if __name__ == "__main__":
    main()