"""
Tahap 2.1-B v5 - Ekstraksi lapisan kulit kepala (konvensi TERVERIFIKASI)
Ground-truth dari 09_mesh_ground_truth.py:
- Indeks .ele BASIS-0 -> JANGAN kurangkan 1 dari sudut tetrahedron.
- Koordinat .node dalam CM -> dikonversi ke mm (x10) saat load,
  sehingga seluruh logika ROI tetap dalam mm dan output siap Geant4.
- Cross-check massa tubuh lolos (01M 9.90 kg vs 10.0; 15F 52.4 vs 50.0).
- Lapisan 12200/12201 hanya ada di HEAD -> silinder verteks bersih.
ROI: silinder r=20 mm x kedalaman 30 mm dari verteks (jangkau beta Sr/Y).
Lapisan (kode atribut .ele keluaran):
  1 = epidermis/permukaan (12200), 2 = basal (12201),
  3 = dermis operasional (jaringan lunak <= 2 mm di bawah kulit),
  4 = cranium cortical (2600), 5 = cranium spongiosa (2700).
Output: D:\\Brachy\\icrp156_work\\<tag>\\layer_*.node/.ele (satuan mm) + JSON.
CLI: python 08_extract_scalp_layers.py [TAG ...]  (tanpa argumen = 6 set)
"""
import sys
import json
import time
from pathlib import Path
import numpy as np

RAW = Path(r"D:\Brachy\icrp156_raw")
WORK = Path(r"D:\Brachy\icrp156_work")
MAPPING = Path(r"D:\Brachy\icrp156_mapping.json")

TARGET_TAGS = ["MRCP_01M", "MRCP_01F", "MRCP_05M", "MRCP_05F", "MRCP_15M", "MRCP_15F"]

CM_TO_MM = 10.0
ID_SKIN_OUT = 12200
ID_SKIN_BASAL = 12201
ID_CRAN_CORT = 2600
ID_CRAN_SPON = 2700
CYL_R_MM = 20.0
CYL_DEPTH_MM = 30.0
CYL_TOP_MM = 5.0
DERMIS_THICK_MM = 2.0
SKIN_MAX_SAMPLE = 20000
CHUNK = 2000

LAYER_NAMES = {1: "epidermis", 2: "basal", 3: "dermis", 4: "cranium_cortical", 5: "cranium_spongiosa"}
EXPECT_CM3 = {1: (0.05, 0.5), 2: (0.02, 0.15), 3: (1.0, 6.0), 4: (1.0, 10.0), 5: (0.5, 10.0)}


def load_mesh(tag):
    hits = sorted(RAW.rglob(f"{tag}.node"))
    if not hits:
        raise FileNotFoundError(f"{tag}.node tidak ditemukan")
    folder = hits[0].parent
    nodes = np.loadtxt(folder / f"{tag}.node", skiprows=1, usecols=(1, 2, 3), dtype=np.float64) * CM_TO_MM
    ele = np.loadtxt(folder / f"{tag}.ele", skiprows=1, usecols=(1, 2, 3, 4, 5), dtype=np.int64)
    return nodes, ele


def centroids_and_volumes(nodes, tets):
    v = nodes[tets]
    a, b, c, d = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
    vol = np.abs(np.einsum("ij,ij->i", a - d, np.cross(b - d, c - d))) / 6.0
    cent = v.mean(axis=1)
    return cent, vol


def cyl_mask(cent, xy0, z_top):
    dxy2 = (cent[:, 0] - xy0[0]) ** 2 + (cent[:, 1] - xy0[1]) ** 2
    return (dxy2 <= CYL_R_MM ** 2) & (cent[:, 2] <= z_top + CYL_TOP_MM) & (cent[:, 2] >= z_top - CYL_DEPTH_MM)


def dermis_shell_mask(cent_soft, skin_cent, thick_mm):
    if len(skin_cent) > SKIN_MAX_SAMPLE:
        idx = np.linspace(0, len(skin_cent) - 1, SKIN_MAX_SAMPLE).astype(int)
        skin_cent = skin_cent[idx]
    out = np.zeros(len(cent_soft), dtype=bool)
    t2 = thick_mm * thick_mm
    for i in range(0, len(cent_soft), CHUNK):
        d = cent_soft[i:i + CHUNK][:, None, :] - skin_cent[None, :, :]
        out[i:i + CHUNK] = (d * d).sum(-1).min(1) <= t2
    return out


def save_subset(out_dir, name, layer_code, nodes, tets):
    used = np.unique(tets.ravel())
    renum = np.zeros(len(nodes), dtype=np.int64)
    renum[used] = np.arange(1, len(used) + 1)
    coords = nodes[used]
    nn = len(used)
    np.savetxt(out_dir / f"layer_{name}.node",
               np.column_stack([np.arange(1, nn + 1), coords]),
               fmt=["%d", "%.4f", "%.4f", "%.4f"],
               header=f"{nn} 3 0 0", comments="")
    ele_out = np.column_stack([np.arange(1, len(tets) + 1), renum[tets], np.full(len(tets), layer_code)])
    np.savetxt(out_dir / f"layer_{name}.ele",
               ele_out, fmt="%d",
               header=f"{len(tets)} 4 1", comments="")
    return nn, len(tets)


def process_tag(tag, mapping_all):
    t0 = time.time()
    print(f"\n--- Ekstraksi v5 {tag} (basis-0, cm->mm) ---")
    nodes, ele = load_mesh(tag)
    tets_all = ele[:, :4]
    ids = ele[:, 4]
    n_all = len(ids)

    bone_ids = set()
    for k, v in mapping_all.get(tag, {}).items():
        low = v.lower()
        if "cortical" in low or "spongiosa" in low:
            bone_ids.add(int(k))
    bone_ids_arr = np.array(sorted(bone_ids), dtype=np.int64)

    idx_out = np.nonzero(ids == ID_SKIN_OUT)[0]
    cent_out, _ = centroids_and_volumes(nodes, tets_all[idx_out])
    z_top = float(cent_out[:, 2].max())
    top_sel = cent_out[:, 2] > z_top - 2.0
    xy0 = cent_out[top_sel][:, :2].mean(axis=0)
    print(f"  verteks: z_top = {z_top:.1f} mm, pusat xy = ({xy0[0]:.1f}, {xy0[1]:.1f}) mm")

    layers = {}
    m1 = np.zeros(n_all, dtype=bool)
    sel = cyl_mask(cent_out, xy0, z_top)
    m1[idx_out[sel]] = True
    layers[1] = m1
    skin_cent_cyl = cent_out[sel]

    idx_bas = np.nonzero(ids == ID_SKIN_BASAL)[0]
    cent_bas, _ = centroids_and_volumes(nodes, tets_all[idx_bas])
    m2 = np.zeros(n_all, dtype=bool)
    selb = cyl_mask(cent_bas, xy0, z_top)
    m2[idx_bas[selb]] = True
    layers[2] = m2
    skin_cent_cyl = np.concatenate([skin_cent_cyl, cent_bas[selb]])

    idx_cort = np.nonzero(ids == ID_CRAN_CORT)[0]
    cent_cort, _ = centroids_and_volumes(nodes, tets_all[idx_cort])
    m4 = np.zeros(n_all, dtype=bool)
    selc = cyl_mask(cent_cort, xy0, z_top)
    m4[idx_cort[selc]] = True
    layers[4] = m4

    idx_spon = np.nonzero(ids == ID_CRAN_SPON)[0]
    cent_spon, _ = centroids_and_volumes(nodes, tets_all[idx_spon])
    m5 = np.zeros(n_all, dtype=bool)
    sels = cyl_mask(cent_spon, xy0, z_top)
    m5[idx_spon[sels]] = True
    layers[5] = m5

    is_skin_id = (ids == ID_SKIN_OUT) | (ids == ID_SKIN_BASAL)
    is_bone = np.isin(ids, bone_ids_arr)
    idx_soft_all = np.nonzero((~is_skin_id) & (~is_bone))[0]
    cent_soft, _ = centroids_and_volumes(nodes, tets_all[idx_soft_all])
    selsoft = cyl_mask(cent_soft, xy0, z_top)
    shell = dermis_shell_mask(cent_soft[selsoft], skin_cent_cyl, DERMIS_THICK_MM)
    m3 = np.zeros(n_all, dtype=bool)
    m3[idx_soft_all[selsoft][shell]] = True
    layers[3] = m3

    out_dir = WORK / tag
    out_dir.mkdir(exist_ok=True)
    row = {"tag": tag, "z_top_mm": z_top, "xy_center_mm": xy0.tolist()}
    for code in [1, 2, 3, 4, 5]:
        mask = layers[code]
        n_tet = int(mask.sum())
        if n_tet == 0:
            print(f"  [PERINGATAN] layer {code} ({LAYER_NAMES[code]}) kosong!")
            row[LAYER_NAMES[code]] = {"n_tet": 0}
            continue
        nn, nt = save_subset(out_dir, LAYER_NAMES[code], code, nodes, tets_all[mask])
        cent, vol = centroids_and_volumes(nodes, tets_all[mask])
        v_cm3 = vol.sum() / 1000.0
        lo_e, hi_e = EXPECT_CM3[code]
        flag = "OK" if lo_e <= v_cm3 <= hi_e else "DI-LUAR-RENTANG"
        row[LAYER_NAMES[code]] = {
            "n_tet": nt,
            "n_node": nn,
            "volume_mm3": float(vol.sum()),
            "zrange_mm": [float(cent[:, 2].min()), float(cent[:, 2].max())],
        }
        print(f"  layer {code} {LAYER_NAMES[code]:>18}: {nt:>8,} tet | vol {v_cm3:>7.3f} cm3 "
              f"(harap {lo_e}-{hi_e}) [{flag}] | z [{cent[:,2].min():6.1f},{cent[:,2].max():6.1f}]")
    print(f"  waktu: {time.time() - t0:.1f} detik")
    return row


def main():
    tags = sys.argv[1:] if len(sys.argv) > 1 else TARGET_TAGS
    print("=" * 70)
    print("TAHAP 2.1-B v5 - EKSTRAKSI KONVENSI TERVERIFIKASI (basis-0, cm->mm)")
    print(f"Set yang akan diproses: {tags}")
    print("=" * 70)
    WORK.mkdir(exist_ok=True)
    mapping_all = {}
    if MAPPING.exists():
        with open(MAPPING, "r") as f:
            mapping_all = json.load(f)

    summary = []
    for tag in tags:
        summary.append(process_tag(tag, mapping_all))

    out_json = WORK / f"extraction_summary_v5_{'_'.join(tags) if len(tags) < 6 else 'all'}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print("\nRingkasan disimpan di:", out_json)
    print("=" * 70)
    print("EKSTRAKSI v5 SELESAI untuk set di atas")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[EKSTRAKSI GAGAL] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)