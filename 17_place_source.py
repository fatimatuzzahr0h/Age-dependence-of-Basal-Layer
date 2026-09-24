"""
Tahap 2.5 - Penempatan sumber Sr-90/Y-90 tervalidasi di verteks tiap fantom
Desain:
- Sumber tervalidasi = stack aplikator lengkap Tahap 1.1/1.2:
  active Sr (r=11.285 mm, t=0.01 mm), substrat Ag 0.1 mm, window PE 1.0 mm,
  barrier PE 10.0 mm; ditempatkan DI ATAS kulit (kulit z<=0, aplikator z>=0).
- Normal lokal n: fit bidang (SVD) patch node epidermis radius 10 mm di
  sekitar apex (node z >= zmax-0.2 mm). Menutup catatan L4 (15M).
- Standoff s = jarak sepanjang n dari apex ke bidang kontak window
  (s=0 -> kontak tangensial).
- Cek otomatis: min gap = s +/- 0.05 mm; penetrasi kulit <= 0; tilt n vs ez;
  smoke-run 200 histori per kombinasi untuk cek overlap Geant4.
Output: D:\\Brachy\\stage25_output\\stage25_placement.json
CLI: python 17_place_source.py
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

TARGET_TAGS = ["MRCP_01M", "MRCP_01F", "MRCP_05M", "MRCP_05F", "MRCP_15M", "MRCP_15F"]
STANDOFFS_MM = [0.0, 0.5, 1.0]
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
    w = w / w.sum()
    return centers, w


def parse_media(tag):
    hits = sorted(RAW.rglob(f"{tag}_media.dat"))
    if not hits:
        raise FileNotFoundError(f"{tag}_media.dat tidak ditemukan")
    meds = {}
    with open(hits[0], "r", errors="ignore") as f:
        for line in f:
            t = line.split()
            if len(t) < 16:
                continue
            try:
                idx = int(t[0])
            except ValueError:
                continue
            try:
                dens = float(t[-1])
                fr = [float(x) for x in t[-14:-1]]
            except ValueError:
                continue
            meds[idx] = dict(name=" ".join(t[1:-14]), density=dens, fractions=fr)
    return meds


def pick_media(meds):
    sel = {}
    for m in meds.values():
        if "skin" in m["name"].lower():
            sel.setdefault("skin", m)
    cort = None
    for m in meds.values():
        low = m["name"].lower()
        if "cranium" in low and "cortical" in low:
            cort = m
            break
    if cort is None:
        for m in meds.values():
            if "cortical" in m["name"].lower():
                cort = m
                break
    if cort is not None:
        sel["cortical"] = cort
    for m in meds.values():
        low = m["name"].lower()
        if "cranium" in low and "spongiosa" in low:
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


def apex_and_normal(nodes):
    zmax = nodes[:, 2].max()
    sel = nodes[:, 2] >= zmax - 0.2
    a = nodes[sel].mean(axis=0)
    d = np.linalg.norm(nodes - a, axis=1)
    patch = nodes[d <= 10.0]
    c = patch.mean(axis=0)
    _, _, vt = np.linalg.svd(patch - c, full_matrices=False)
    n = vt[2].copy()
    if n[2] < 0:
        n = -n
    return a, n


def rot_z_to(n):
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, n)
    c = float(np.dot(z, n))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + (vx @ vx) * (1.0 / (1.0 + c))


def gap_check(nodes, a, n, s):
    c_s = a + n * s
    proj = (nodes - c_s) @ n
    return float(-proj.max()), float(proj.max())


def build_combo(tag, s_mm, apex, nrm, meds_sel):
    sim = gate.Simulation()
    sim.number_of_threads = 1
    sim.visu = False
    sim.progress_bar = False
    sim.output_dir = str(OUT / tag / f"s{s_mm:.1f}")
    mm = gate.g4_units.mm
    MeV = gate.g4_units.MeV

    sim.world.size = [400 * mm, 400 * mm, 400 * mm]
    sim.world.material = "G4_AIR"

    made = {}
    for layer in LAYERS:
        mk = MAT_KEY[layer]
        if mk not in made:
            m = meds_sel[mk]
            mname = f"ICRP156_{tag}_{mk}"
            define_material(sim, mname, m["density"], m["fractions"])
            made[mk] = mname
        v = sim.add_volume("TesselatedVolume", name=f"{tag}_{layer}")
        v.file_name = str(WORK / tag / "stl" / f"layer_{layer}.stl")
        v.material = made[mk]
        v.translation = [0 * mm, 0 * mm, 0 * mm]

    R = rot_z_to(nrm)
    stack = [
        ("window", 0.5, 0.5, 12.0, "G4_POLYETHYLENE", 0.01),
        ("substrate", 1.05, 0.05, 11.285, "G4_Ag", 0.01),
        ("active", 1.105, 0.005, 11.285, "G4_Sr", 0.01),
        ("barrier", 6.11, 5.0, 13.0, "G4_POLYETHYLENE", 0.1),
    ]
    for name, tz, dz, rmax, mat, cut in stack:
        v = sim.add_volume("Tubs", name=f"{tag}_{name}")
        v.rmin = 0
        v.rmax = rmax * mm
        v.dz = dz * mm
        t = apex + nrm * (s_mm + tz)
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
    tp = apex + nrm * (s_mm + 1.105)
    src.position.translation = [tp[0] * mm, tp[1] * mm, tp[2] * mm]
    try:
        src.position.rotation = R
    except Exception:
        print("  [WARN] source.position.rotation tidak tersedia:",
              [a2 for a2 in dir(src.position) if not a2.startswith("_")])
        raise
    src.direction.type = "iso"

    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option4"
    sim.physics_manager.enable_decay = False
    sim.physics_manager.set_production_cut("world", "all", 1 * mm)
    for layer in LAYERS:
        sim.physics_manager.set_production_cut(f"{tag}_{layer}", "all", 0.01 * mm)
    return sim


def main():
    OUT.mkdir(exist_ok=True)
    print("=" * 78)
    print("TAHAP 2.5 - PENEMPATAN SUMBER TERVERIDASI (6 set x 3 standoff)")
    print("=" * 78)

    placements = {}
    for tag in TARGET_TAGS:
        nodes = np.loadtxt(WORK / tag / "layer_epidermis.node",
                           skiprows=1, usecols=(1, 2, 3), dtype=np.float64)
        a, n = apex_and_normal(nodes)
        tilt = np.degrees(np.arccos(np.clip(n[2], -1, 1)))
        placements[tag] = dict(apex=a.tolist(), normal=n.tolist(),
                               tilt_deg=float(tilt), nodes=nodes)
        print(f"{tag}: apex=({a[0]:7.2f},{a[1]:7.2f},{a[2]:6.2f}) mm | "
              f"n=({n[0]:+.3f},{n[1]:+.3f},{n[2]:+.3f}) | tilt {tilt:5.2f} deg")

    results = []
    for tag in TARGET_TAGS:
        meds_sel = pick_media(parse_media(tag))
        a = np.array(placements[tag]["apex"])
        n = np.array(placements[tag]["normal"])
        nodes = placements[tag].pop("nodes")
        for s in STANDOFFS_MM:
            mingap, maxpen = gap_check(nodes, a, n, s)
            sim = build_combo(tag, s, a, n, meds_sel)
            t0 = time.time()
            sim.run(start_new_process=True)
            dt = time.time() - t0
            ok_gap = abs(mingap - s) <= 0.05
            ok_pen = maxpen <= 0.001
            print(f"  {tag} s={s:.1f} mm: min gap {mingap:6.3f} mm [{ 'OK' if ok_gap else 'CEK!' }] | "
                  f"max pen {maxpen:7.3f} mm [{ 'OK' if ok_pen else 'CEK!' }] | run {dt:5.1f} s")
            results.append(dict(tag=tag, standoff_mm=s,
                                min_gap_mm=mingap, max_penetration_mm=maxpen,
                                tilt_deg=placements[tag]["tilt_deg"],
                                gap_ok=bool(ok_gap), pen_ok=bool(ok_pen), run_ok=True))

    with open(OUT / "stage25_placement.json", "w") as f:
        json.dump(results, f, indent=2)
    n_bad = sum(1 for r in results if not (r["gap_ok"] and r["pen_ok"] and r["run_ok"]))
    print("\nRingkasan:", OUT / "stage25_placement.json")
    print(f"Kombinasi lulus cek penempatan : {len(results) - n_bad}/{len(results)}")
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