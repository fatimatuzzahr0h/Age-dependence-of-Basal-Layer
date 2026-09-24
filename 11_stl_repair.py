"""
Tahap 2.2b - Perbaikan topologis subset tet + regenerasi STL
Strategi: iterative tet deletion. Tet yang bertetangga dengan edge buruk
(edge yang dipakai != 2 muka batas) dibuang; maksimal 3 ronde;
kehilangan volume harus <= 2%.
Lapisan diperbaiki: epidermis, basal, cranium_cortical, cranium_spongiosa.
Lapisan dermis: DIBEBASKAN (visual-only; skoring via aktor terstruktur di Tahap 3).
Output: subset .node/.ele tertimpa + STL biner baru + stl_summary_repaired.json
CLI: python 11_stl_repair.py [TAG ...]  (tanpa argumen = 6 set)
"""
import sys
import json
import time
from pathlib import Path
import numpy as np

WORK = Path(r"D:\Brachy\icrp156_work")
TARGET_TAGS = ["MRCP_01M", "MRCP_01F", "MRCP_05M", "MRCP_05F", "MRCP_15M", "MRCP_15F"]
REPAIR_LAYERS = ["epidermis", "basal", "cranium_cortical", "cranium_spongiosa"]
MAX_ROUNDS = 3
MAX_VOL_LOSS = 0.02


def load_layer(src, name):
    nodes = np.loadtxt(src / f"layer_{name}.node", skiprows=1, usecols=(1, 2, 3), dtype=np.float64)
    ele = np.loadtxt(src / f"layer_{name}.ele", skiprows=1, usecols=(1, 2, 3, 4), dtype=np.int64)
    return nodes, ele - 1


def tet_volumes(nodes, tets):
    v = nodes[tets]
    a, b, c, d = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
    return np.abs(np.einsum("ij,ij->i", a - d, np.cross(b - d, c - d))) / 6.0


def boundary_and_bad_edges(tets):
    f = np.empty((len(tets), 4, 3), dtype=np.int64)
    f[:, 0] = tets[:, [0, 2, 1]]
    f[:, 1] = tets[:, [0, 1, 3]]
    f[:, 2] = tets[:, [0, 3, 2]]
    f[:, 3] = tets[:, [1, 2, 3]]
    flat = f.reshape(-1, 3)
    sorted_f = np.sort(flat, axis=1)
    uniq, inv, cnt = np.unique(sorted_f, axis=0, return_inverse=True, return_counts=True)
    bfaces = flat[cnt[inv] == 1]
    e = np.concatenate([bfaces[:, [0, 1]], bfaces[:, [1, 2]], bfaces[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)
    uniq_e, cnt_e = np.unique(e, axis=0, return_counts=True)
    bad = uniq_e[cnt_e != 2]
    return bfaces, bad


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


def save_subset(src, name, nodes, tets):
    used = np.unique(tets.ravel())
    renum = np.zeros(len(nodes), dtype=np.int64)
    renum[used] = np.arange(1, len(used) + 1)
    nn = len(used)
    np.savetxt(src / f"layer_{name}.node",
               np.column_stack([np.arange(1, nn + 1), nodes[used]]),
               fmt=["%d", "%.4f", "%.4f", "%.4f"],
               header=f"{nn} 3 0 0", comments="")
    ele_out = np.column_stack([np.arange(1, len(tets) + 1), renum[tets], np.ones(len(tets), dtype=np.int64)])
    np.savetxt(src / f"layer_{name}.ele", ele_out, fmt="%d",
               header=f"{len(tets)} 4 1", comments="")


def repair_layer(nodes, tets):
    vol0 = tet_volumes(nodes, tets).sum()
    history = []
    for rnd in range(MAX_ROUNDS):
        bfaces, bad = boundary_and_bad_edges(tets)
        history.append(int(len(bad)))
        if len(bad) == 0:
            return tets, history, True
        kill = np.zeros(len(tets), dtype=bool)
        for a, b in bad:
            kill |= (tets == a).any(axis=1) & (tets == b).any(axis=1)
        tets = tets[~kill]
        if len(tets) == 0:
            return tets, history, False
    bfaces, bad = boundary_and_bad_edges(tets)
    history.append(int(len(bad)))
    vol1 = tet_volumes(nodes, tets).sum()
    loss = 1.0 - vol1 / vol0 if vol0 > 0 else 1.0
    ok = (len(bad) == 0) and (loss <= MAX_VOL_LOSS)
    return tets, history, ok


def process_tag(tag):
    src = WORK / tag
    dst = src / "stl"
    rows = []
    print(f"\n--- Perbaikan {tag} ---")
    for name in REPAIR_LAYERS:
        nodes, tets = load_layer(src, name)
        n0 = len(tets)
        vol0 = tet_volumes(nodes, tets).sum()
        tets2, hist, ok = repair_layer(nodes, tets)
        vol1 = tet_volumes(nodes, tets2).sum() if len(tets2) else 0.0
        loss = 1.0 - vol1 / vol0 if vol0 > 0 else 1.0
        if len(tets2) > 0:
            save_subset(src, name, nodes, tets2)
            bfaces, bad = boundary_and_bad_edges(tets2)
            vol_surf, area = surface_metrics(nodes, bfaces)
            write_binary_stl(dst / f"layer_{name}.stl", nodes, bfaces)
            n_tri = len(bfaces)
        else:
            n_tri = 0
            area = 0.0
        status = "WATERTIGHT" if ok else f"GAGAL (bad={hist[-1]})"
        print(f"  {name:>18}: tet {n0:>6,} -> {len(tets2):>6,} | bad edges {hist} | "
              f"vol loss {loss*100:.2f}% | {n_tri:>6,} tri | {status}")
        rows.append({"tag": tag, "layer": name, "n_tet_before": n0, "n_tet_after": int(len(tets2)),
                     "bad_edge_history": hist, "vol_loss_frac": float(loss),
                     "n_tri": int(n_tri), "watertight": bool(ok)})
    return rows


def main():
    tags = sys.argv[1:] if len(sys.argv) > 1 else TARGET_TAGS
    print("=" * 70)
    print("TAHAP 2.2b - PERBAIKAN TOPOLOGIS + REGENERASI STL")
    print(f"Set: {tags} | dermis DIBEBASKAN (visual-only)")
    print("=" * 70)
    rows = []
    for tag in tags:
        rows.extend(process_tag(tag))
    out = WORK / "stl_summary_repaired.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    n_fail = sum(1 for r in rows if not r["watertight"])
    print("\nRingkasan:", out)
    print(f"Lapisan diperbaiki watertight : {len(rows) - n_fail}/{len(rows)}")
    print(f"Lapisan masih gagal           : {n_fail}")
    print("=" * 70)
    print("TAHAP 2.2b SELESAI")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[PERBAIKAN GAGAL] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)