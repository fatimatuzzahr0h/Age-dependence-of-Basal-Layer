"""
Tahap 2.2c - Repair tertarget untuk lapisan yang gagal kaskade pinch
Alur: (1) regenerasi subset dari ekstraksi v5 (jalankan 08 dulu untuk tag tsb),
(2) buang tetrahedra duplikat, (3) buang tet pada muka bermultiplisitas >= 3,
(4) pertahankan komponen face-connected terbesar (union-find),
(5) cek watertight + vol loss <= 5% (batas longgar: layer sekunder),
(6) tulis ulang subset + STL biner.
CLI: python 12_repair_targeted.py MRCP_15M cranium_spongiosa
"""
import sys
import json
from pathlib import Path
import numpy as np

WORK = Path(r"D:\Brachy\icrp156_work")
MAX_LOSS = 0.05


def load_layer(src, name):
    nodes = np.loadtxt(src / f"layer_{name}.node", skiprows=1, usecols=(1, 2, 3), dtype=np.float64)
    ele = np.loadtxt(src / f"layer_{name}.ele", skiprows=1, usecols=(1, 2, 3, 4), dtype=np.int64)
    return nodes, ele - 1


def tet_vols(nodes, tets):
    v = nodes[tets]
    a, b, c, d = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
    return np.abs(np.einsum("ij,ij->i", a - d, np.cross(b - d, c - d))) / 6.0


def face_table(tets):
    f = np.empty((len(tets), 4, 3), dtype=np.int64)
    f[:, 0] = tets[:, [0, 2, 1]]
    f[:, 1] = tets[:, [0, 1, 3]]
    f[:, 2] = tets[:, [0, 3, 2]]
    f[:, 3] = tets[:, [1, 2, 3]]
    flat = f.reshape(-1, 3)
    srt = np.sort(flat, axis=1)
    uniq, inv, cnt = np.unique(srt, axis=0, return_inverse=True, return_counts=True)
    return flat, inv, cnt


def boundary_and_bad_edges(tets):
    flat, inv, cnt = face_table(tets)
    bfaces = flat[cnt[inv] == 1]
    e = np.concatenate([bfaces[:, [0, 1]], bfaces[:, [1, 2]], bfaces[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)
    uniq_e, cnt_e = np.unique(e, axis=0, return_counts=True)
    return bfaces, uniq_e[cnt_e != 2]


def largest_component(tets):
    flat, inv, cnt = face_table(tets)
    n = len(tets)
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    face_to_tets = {}
    for fi, ti in enumerate(inv):
        if cnt[ti] == 2:
            face_to_tets.setdefault(ti, []).append(fi // 4)
    for ti, tlist in face_to_tets.items():
        if len(tlist) == 2:
            ra, rb = find(tlist[0]), find(tlist[1])
            if ra != rb:
                parent[ra] = rb
    roots = np.array([find(i) for i in range(n)])
    vols = tet_vols_nodes_cache if False else None
    return roots


def process(tag, layer):
    src = WORK / tag
    nodes, tets = load_layer(src, layer)
    vol0 = tet_vols(nodes, tets).sum()
    print(f"--- {tag} / {layer}: {len(tets)} tet awal, vol {vol0:.1f} mm3 ---")

    key = np.sort(tets, axis=1)
    uniq, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    dup = cnt[inv] > 1
    if dup.any():
        print(f"  buang {int(dup.sum())} tet duplikat")
        tets = tets[~dup]

    for rnd in range(3):
        flat, invf, cntf = face_table(tets)
        bad_face_ids = np.nonzero(cntf >= 3)[0]
        if len(bad_face_ids) == 0:
            break
        bad_per_face = cntf[invf]
        kill = bad_per_face >= 3
        print(f"  ronde {rnd}: buang {int(kill.sum())} tet pada muka multiplisitas>=3")
        tets = tets[~kill]
        if len(tets) == 0:
            break

    if len(tets) > 0:
        roots = largest_component(tets)
        ur, counts = np.unique(roots, return_counts=True)
        if len(ur) > 1:
            vols = tet_vols(nodes, tets)
            comp_vol = np.zeros(len(ur))
            for i, r in enumerate(ur):
                comp_vol[i] = vols[roots == r].sum()
            keep_root = ur[np.argmax(comp_vol)]
            dropped = int((roots != keep_root).sum())
            print(f"  komponen: {len(ur)} -> pertahankan terbesar, buang {dropped} tet")
            tets = tets[roots == keep_root]

    bfaces, bad = boundary_and_bad_edges(tets)
    vol1 = tet_vols(nodes, tets).sum()
    loss = 1.0 - vol1 / vol0
    ok = (len(bad) == 0) and (loss <= MAX_LOSS)
    print(f"  akhir: {len(tets)} tet | bad edges {len(bad)} | vol loss {loss*100:.2f}% | "
          f"{'WATERTIGHT-OK' if ok else 'MASIH GAGAL'}")
    if not ok:
        print("  => gunakan jalur pengecualian: skor sumsum 15M via aktor terstruktur (Tahap 3).")
        return False

    used = np.unique(tets.ravel())
    renum = np.zeros(len(nodes), dtype=np.int64)
    renum[used] = np.arange(1, len(used) + 1)
    np.savetxt(src / f"layer_{layer}.node",
               np.column_stack([np.arange(1, len(used) + 1), nodes[used]]),
               fmt=["%d", "%.4f", "%.4f", "%.4f"],
               header=f"{len(used)} 3 0 0", comments="")
    ele_out = np.column_stack([np.arange(1, len(tets) + 1), renum[tets], np.ones(len(tets), dtype=np.int64)])
    np.savetxt(src / f"layer_{layer}.ele", ele_out, fmt="%d",
               header=f"{len(tets)} 4 1", comments="")

    v0 = nodes[tets[:, 0]].astype(np.float32)
    v1 = nodes[tets[:, 1]].astype(np.float32)
    v2 = nodes[tets[:, 2]].astype(np.float32)
    nrm = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(nrm, axis=1)
    ln[ln == 0] = 1.0
    nrm = (nrm / ln[:, None]).astype(np.float32)
    tri = np.zeros(len(bfaces), dtype=[("n", np.float32, (3,)), ("v", np.float32, (3, 3)), ("attr", np.uint16)])
    fb0 = nodes[bfaces[:, 0]].astype(np.float32)
    fb1 = nodes[bfaces[:, 1]].astype(np.float32)
    fb2 = nodes[bfaces[:, 2]].astype(np.float32)
    nb = np.cross(fb1 - fb0, fb2 - fb0)
    lb = np.linalg.norm(nb, axis=1)
    lb[lb == 0] = 1.0
    tri["n"] = (nb / lb[:, None]).astype(np.float32)
    tri["v"][:, 0] = fb0
    tri["v"][:, 1] = fb1
    tri["v"][:, 2] = fb2
    stl_path = src / "stl" / f"layer_{layer}.stl"
    with open(stl_path, "wb") as fh:
        fh.write(b"\0" * 80)
        fh.write(np.uint32(len(bfaces)).tobytes())
        fh.write(tri.tobytes())
    print(f"  STL ditulis: {stl_path}")
    return True


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "MRCP_15M"
    layer = sys.argv[2] if len(sys.argv) > 2 else "cranium_spongiosa"
    try:
        process(tag, layer)
    except Exception as e:
        print(f"[GAGAL] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)