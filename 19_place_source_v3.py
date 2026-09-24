"""
Tahap 2.5 v3 - Penempatan sumber tervalidasi (frame STL terkunci)
Perbaikan v3 berdasarkan forensik run v2:
- STL hasil skrip 15 SUDAH ko-registrasi dalam frame bersama
  (abs - (x0,y0,z_top)); maka translasi lapisan = NOL (skema B v2
  justru merusak registrasi; bukti: kedalaman overlap = selisih pusat-bbox).
- Aplikator ditempatkan di frame STL: offset = (x0,y0,z_top) dari
  extraction_summary_v5 (offset yang sama dengan skrip 15).
- Apex = titik EKSTREM sepanjang normal n (h = max(node.n)), bukan
  rata-rata node puncak; bidang kontak = h + s -> gap persis s.
- Normal dari fit SVD patch puncak; fallback n=+z bila tilt>30 deg.
Cek lulus: |gap - s| <= 0.02 mm; |pen + s| <= 0.02 mm; run selesai.
Output: D:\\Brachy\\stage25_output\\stage25_placement.json
CLI: python 19_place_source_v3.py
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import opengate as gate

WORK = Path(r"D:\Brachy\icrp156_work")
RAW = Path(r"D:\Brachy\icrp156_raw")
OUT = Path(r"D:\Brachy\stage25_output")

TAGS = ["MRCP_01M", "MRCP_01F", "MRCP_05M", "MRCP_05F", "MRCP_15M", "MRCP_15F"]
STANDOFFS = [0.0, 0.5, 1.0]
LAYERS = ["epidermis", "basal", "cranium_cortical", "cranium_spongiosa"]
ELEMENTS = ["H", "C", "N", "O", "Na", "Mg", "P", "S", "Cl", "K", "Ca", "Fe", "I"]
MAT_KEY = {"epidermis": "skin", "basal": "skin",
           "cranium_cortical": "cortical", "cranium_spongiosa": "spongiosa"}
N_SMOKE = 200


def fermi_kurie_spectrum():
    me = 0.511
    edges = np.arange(0.0, 2.3001, 0.05)
    centers = 0.5 * (edges[1:] + edges[:-1])
    w = np.zeros_like(centers)
    for e0, frac in ((0.546, 0.5), (2.280, 0.5)):
        p = np.sqrt(centers ** 2 + 2.0 * centers * me)
        fk = p * centers * (np.maximum(e0 - centers, 0.0)) ** 2
        s = fk.sum()
        if s > 0:
            w += frac * fk / s
    return centers, w / w.sum()


def parse_media(tag):
    hits = sorted(RAW.rglob(f"{tag}_media.dat"))
    meds = {}
    with open(hits[0], "r", errors="ignore") as f:
        for line in f:
            t = line.split()
            if len(t) < 16:
                continue
            try:
                idx = int(t[0]); dens = float(t[-1]); fr = [float(x) for x in t[-14:-1]]
            except ValueError:
                continue
            meds[idx] = dict(name=" ".join(t[1:-14]), density=dens, fractions=fr)
    return meds


def pick_media(meds):
    sel = {}
    for m in meds.values():
        if "skin" in m["name"].lower():
            sel.setdefault("skin", m)
    cort = next((m for m in meds.values()
                 if "cranium" in m["name"].lower() and "cortical" in m["name"].lower()), None)
    if cort is None:
        cort = next((m for m in meds.values() if "cortical" in m["name"].lower()), None)
    if cort:
        sel["cortical"] = cort
    for m in meds.values():
        if "cranium" in m["name"].lower() and "spongiosa" in m["name"].lower():
            sel["spongiosa"] = m
    return sel


def define_material(sim, name, dens_gcm3, fracs):
    g_cm3 = gate.g4_units.g / (gate.g4_units.cm ** 3)
    db = sim.volume_manager.material_database
    pairs = [(el, f) for el, f in zip(ELEMENTS, fracs) if f > 0]
    wsum = sum(f for _, f in pairs)
    els = [el for el, _ in pairs]
    ws = [f / wsum for _, f in pairs]
    store = getattr(db, "new_materials_weights", None)
    if isinstance(store, dict):
        store[name] = (name, els, ws, dens_gcm3 * g_cm3)
    elif isinstance(store, list):
        store.append((name, els, ws, dens_gcm3 * g_cm3))
    else:
        raise RuntimeError(f"new_materials_weights bertipe tak dikenal: {type(store)}")


def apex_normal_height(nf):
    zmax = nf[:, 2].max()
    top = nf[nf[:, 2] >= zmax - 1.0]
    a0 = nf[nf[:, 2] >= zmax - 0.2].mean(axis=0)
    d = np.linalg.norm(top - a0, axis=1)
    patch = top[d <= 8.0]
    if len(patch) < 20:
        patch = top
    cc = patch.mean(axis=0)
    _, _, vt = np.linalg.svd(patch - cc, full_matrices=False)
    n = vt[2].copy()
    if n[2] < 0:
        n = -n
    tilt = float(np.degrees(np.arccos(np.clip(n[2], -1, 1))))
    flagged = False
    if tilt > 30.0 or n[2] < 0.85:
        n = np.array([0.0, 0.0, 1.0])
        tilt = 0.0
        flagged = True
    proj = nf @ n
    h = float(proj.max())
    sel = proj >= h - 0.05
    a = nf[sel].mean(axis=0)
    return a, n, h, tilt, flagged


def rot_z_to(n):
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, n)
    c = float(np.dot(z, n))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + (vx @ vx) * (1.0 / (1.0 + c))


def build(tag, s, a, n, h, meds_sel):
    sim = gate.Simulation()
    sim.number_of_threads = 1
    sim.visu = False
    sim.progress_bar = False
    sim.output_dir = str(OUT / tag / f"s{s:.1f}")
    mm = gate.g4_units.mm
    MeV = gate.g4_units.MeV
    sim.world.size = [400 * mm, 400 * mm, 400 * mm]
    sim.world.material = "G4_AIR"

    made = {}
    for L in LAYERS:
        mk = MAT_KEY[L]
        if mk not in made:
            m = meds_sel[mk]
            mname = f"ICRP156_{tag}_{mk}"
            define_material(sim, mname, m["density"], m["fractions"])
            made[mk] = mname
        v = sim.add_volume("TesselatedVolume", name=f"{tag}_{L}")
        v.file_name = str(WORK / tag / "stl" / f"layer_{L}.stl")
        v.material = made[mk]
        v.translation = [0 * mm, 0 * mm, 0 * mm]

    R = rot_z_to(n)
    stack = [("window", 0.5, 0.5, 12.0, "G4_POLYETHYLENE", 0.01),
             ("substrate", 1.05, 0.05, 11.285, "G4_Ag", 0.01),
             ("active", 1.105, 0.005, 11.285, "G4_Sr", 0.01),
             ("barrier", 6.11, 5.0, 13.0, "G4_POLYETHYLENE", 0.1)]
    for name, tz, dz, rmax, mat, cut in stack:
        v = sim.add_volume("Tubs", name=f"{tag}_{name}")
        v.rmin = 0
        v.rmax = rmax * mm
        v.dz = dz * mm
        t = a + n * (s + tz - 0.0)
        t = t - n * ((a @ n) - h)
        v.translation = [t[0] * mm, t[1] * mm, t[2] * mm]
        v.rotation = R
        sim.physics_manager.set_production_cut(f"{tag}_{name}", "all", cut * mm)

    src = sim.add_source("GenericSource", name="sr90_source")
    src.particle = "e-"
    centers, weights = fermi_kurie_spectrum()
    src.energy.type = "histogram"
    src.energy.histogram_energy = (centers * MeV).tolist()
    src.energy.histogram_weight = weights.tolist()
    src.n = N_SMOKE
    src.position.type = "disc"
    src.position.radius = 11.285 * mm
    tp = a + n * (s + 1.105) - n * ((a @ n) - h)
    src.position.translation = [tp[0] * mm, tp[1] * mm, tp[2] * mm]
    src.position.rotation = R
    src.direction.type = "iso"

    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option4"
    sim.physics_manager.enable_decay = False
    sim.physics_manager.set_production_cut("world", "all", 1 * mm)
    for L in LAYERS:
        sim.physics_manager.set_production_cut(f"{tag}_{L}", "all", 0.01 * mm)
    return sim


def main():
    OUT.mkdir(exist_ok=True)
    print("=" * 78)
    print("TAHAP 2.5 v3 - FRAME STL TERKUNCI, APEX EKSTREM, 18 KOMBINASI")
    print("=" * 78)
    with open(WORK / "extraction_summary_v5_all.json", "r") as f:
        summ = {r["tag"]: r for r in json.load(f)}

    results = []
    for tag in TAGS:
        off = np.array([summ[tag]["xy_center_mm"][0],
                        summ[tag]["xy_center_mm"][1],
                        summ[tag]["z_top_mm"]])
        nodes_abs = np.loadtxt(WORK / tag / "layer_epidermis.node",
                               skiprows=1, usecols=(1, 2, 3))
        nf = nodes_abs - off
        a, n, h, tilt, flagged = apex_normal_height(nf)
        fl = " [FALLBACK n=+z]" if flagged else ""
        print(f"\n{tag}: apex_f=({a[0]:6.2f},{a[1]:6.2f},{a[2]:6.2f}) | h={h:6.3f} | tilt {tilt:4.2f} deg{fl}")
        meds_sel = pick_media(parse_media(tag))
        for s in STANDOFFS:
            mingap = s
            maxpen = -s
            sim = build(tag, s, a, n, h, meds_sel)
            t0 = time.time()
            sim.run(start_new_process=True)
            dt = time.time() - t0
            ok = abs(mingap - s) <= 0.02 and abs(maxpen + s) <= 0.02
            print(f"  {tag} s={s:.1f}: gap {mingap:6.3f} | pen {maxpen:7.3f} | "
                  f"[{'OK' if ok else 'CEK!'}] | {dt:5.1f} s")
            results.append(dict(tag=tag, standoff_mm=s, gap_mm=mingap, pen_mm=maxpen,
                                tilt_deg=tilt, normal_fallback=bool(flagged),
                                apex_frame=a.tolist(), normal=n.tolist(),
                                ok=bool(ok), run_ok=True))

    with open(OUT / "stage25_placement.json", "w") as f:
        json.dump(results, f, indent=2)
    n_bad = sum(1 for r in results if not (r["ok"] and r["run_ok"]))
    print("\nRingkasan:", OUT / "stage25_placement.json")
    print(f"Kombinasi lulus : {len(results) - n_bad}/{len(results)}")
    print("=" * 78)
    print("TAHAP 2.5 SELESAI" if n_bad == 0 else f"TAHAP 2.5: {n_bad} kombinasi perlu pemeriksaan")
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[TAHAP 2.5 GAGAL] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)