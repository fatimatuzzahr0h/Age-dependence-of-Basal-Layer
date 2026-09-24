"""
Tahap 3.1 - Validasi mekanisme skoring DoseActor pada TesselatedVolume
Uji diskriminatif: basal & epidermis berbagi bounding box yang hampir
identik. Jika DoseActor 1x1x1 terbatas-volume -> edep_basal/edep_epi
~ 0.05-0.25; jika berbasis bbox -> rasio ~ 1.0.
Sekalian: aktor silinder profil kedalaman (250 bin x 0.1 mm) menempel
ke world (rotasi = normal lokal), BUKAN volume fisik -> bebas overlap.
Konfigurasi uji: MRCP_01M, standoff 0 mm, N = 2e6 histori.
Output: D:\\Brachy\\stage3_output\\stage30_mechanism_test.json
CLI: python 20_score_mechanism_test.py
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import opengate as gate

WORK = Path(r"D:\Brachy\icrp156_work")
RAW = Path(r"D:\Brachy\icrp156_raw")
OUT = Path(r"D:\Brachy\stage3_output")

TAG = "MRCP_01M"
STANDOFF = 0.0
N_HIST = 2_000_000
LAYERS = ["epidermis", "basal", "cranium_cortical", "cranium_spongiosa"]
ELEMENTS = ["H", "C", "N", "O", "Na", "Mg", "P", "S", "Cl", "K", "Ca", "Fe", "I"]
MAT_KEY = {"epidermis": "skin", "basal": "skin",
           "cranium_cortical": "cortical", "cranium_spongiosa": "spongiosa"}
MEV_TO_J = 1.602176634e-13


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


def read_image(path, retries=5):
    import itk
    for _ in range(retries):
        if Path(path).exists():
            break
        time.sleep(1)
    return itk.array_from_image(itk.imread(str(path))).astype(float)


def main():
    OUT.mkdir(exist_ok=True)
    print("=" * 78)
    print(f"TAHAP 3.1 - UJI MEKANISME SKORING | {TAG} standoff {STANDOFF} mm | N={N_HIST:.0e}")
    print("=" * 78)

    with open(WORK / "extraction_summary_v5_all.json", "r") as f:
        summ = {r["tag"]: r for r in json.load(f)}
    off = np.array([summ[TAG]["xy_center_mm"][0], summ[TAG]["xy_center_mm"][1], summ[TAG]["z_top_mm"]])

    geo = {}
    for L in LAYERS:
        nf = np.loadtxt(WORK / TAG / f"layer_{L}.node", skiprows=1, usecols=(1, 2, 3)) - off
        geo[L] = dict(center=(nf.min(0) + nf.max(0)) / 2.0, extent=nf.max(0) - nf.min(0))

    nf_epi = np.loadtxt(WORK / TAG / "layer_epidermis.node", skiprows=1, usecols=(1, 2, 3)) - off
    a, n, h, tilt, flagged = apex_normal_height(nf_epi)
    R = rot_z_to(n)
    print(f"apex=({a[0]:.2f},{a[1]:.2f},{a[2]:.2f}) | tilt {tilt:.2f} deg")

    sim = gate.Simulation()
    sim.number_of_threads = 1
    sim.visu = False
    sim.progress_bar = True
    sim.output_dir = str(OUT)
    mm = gate.g4_units.mm
    MeV = gate.g4_units.MeV
    sim.world.size = [400 * mm, 400 * mm, 400 * mm]
    sim.world.material = "G4_AIR"

    meds_sel = pick_media(parse_media(TAG))
    made = {}
    for L in LAYERS:
        mk = MAT_KEY[L]
        if mk not in made:
            m = meds_sel[mk]
            mname = f"ICRP156_{TAG}_{mk}"
            define_material(sim, mname, m["density"], m["fractions"])
            made[mk] = (mname, m)
        v = sim.add_volume("TesselatedVolume", name=f"{TAG}_{L}")
        v.file_name = str(WORK / TAG / "stl" / f"layer_{L}.stl")
        v.material = made[mk][0]
        v.translation = [0 * mm, 0 * mm, 0 * mm]

    s = STANDOFF
    stack = [("window", 0.5, 0.5, 12.0, "G4_POLYETHYLENE", 0.01),
             ("substrate", 1.05, 0.05, 11.285, "G4_Ag", 0.01),
             ("active", 1.105, 0.005, 11.285, "G4_Sr", 0.01),
             ("barrier", 6.11, 5.0, 13.0, "G4_POLYETHYLENE", 0.1)]
    for name, tz, dz, rmax, mat, cut in stack:
        v = sim.add_volume("Tubs", name=f"{TAG}_{name}")
        v.rmin = 0
        v.rmax = rmax * mm
        v.dz = dz * mm
        t = a + n * (s + tz) - n * ((a @ n) - h)
        v.translation = [t[0] * mm, t[1] * mm, t[2] * mm]
        v.rotation = R
        sim.physics_manager.set_production_cut(f"{TAG}_{name}", "all", cut * mm)

    src = sim.add_source("GenericSource", name="sr90_source")
    src.particle = "e-"
    centers, weights = fermi_kurie_spectrum()
    src.energy.type = "histogram"
    src.energy.histogram_energy = (centers * MeV).tolist()
    src.energy.histogram_weight = weights.tolist()
    src.n = N_HIST
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
        sim.physics_manager.set_production_cut(f"{TAG}_{L}", "all", 0.01 * mm)

    # Actor lapisan: 1 voxel = bbox lapisan (uji mekanisme)
    for L in LAYERS:
        d = sim.add_actor("DoseActor", f"dose_{L}")
        d.attached_to = f"{TAG}_{L}"
        d.size = [1, 1, 1]
        e = geo[L]["extent"]
        c = geo[L]["center"]
        d.spacing = [max(e[0], 0.01) * mm, max(e[1], 0.01) * mm, max(e[2], 0.01) * mm]
        d.translation = [c[0] * mm, c[1] * mm, c[2] * mm]
        d.edep.active = True
        d.dose.active = True
        d.dose_uncertainty.active = True
        d.edep.output_filename = f"edep_{L}.mhd"
        d.dose.output_filename = f"dose_{L}.mhd"
        d.dose_uncertainty.output_filename = f"unc_{L}.mhd"

    # Actor silinder profil kedalaman: menempel world, bukan volume fisik
    cyl = sim.add_actor("DoseActor", "cyl_profile")
    cyl.attached_to = "world"
    cyl.size = [1, 1, 250]
    cyl.spacing = [40 * mm, 40 * mm, 0.1 * mm]
    ccenter = a + n * (-10.5) - n * ((a @ n) - h)
    cyl.translation = [ccenter[0] * mm, ccenter[1] * mm, ccenter[2] * mm]
    cyl.rotation = R
    cyl.edep.active = True
    cyl.dose.active = True
    cyl.dose_uncertainty.active = True
    cyl.edep.output_filename = "cyl_edep.mhd"
    cyl.dose.output_filename = "cyl_dose.mhd"
    cyl.dose_uncertainty.output_filename = "cyl_unc.mhd"

    t0 = time.time()
    sim.run(start_new_process=True)
    dt = time.time() - t0
    print(f"run selesai {dt:.1f} s")

    results = dict(tag=TAG, standoff_mm=s, n_hist=N_HIST, layers={})
    print("\n--- edep per lapisan (MeV total) & dosis per histori ---")
    for L in LAYERS:
        edep = read_image(OUT / f"edep_{L}.mhd").sum()
        unc = read_image(OUT / f"unc_{L}.mhd").flatten()[0]
        vol = summ[TAG][L]["volume_mm3"]
        dens = meds_sel[MAT_KEY[L]]["density"]
        mass_kg = vol * dens / 1000.0 / 1000.0
        dose_gy = edep * MEV_TO_J / mass_kg
        results["layers"][L] = dict(edep_MeV=float(edep), unc_rel=float(unc),
                                    mass_g=vol * dens / 1000.0,
                                    dose_Gy_per_hist=float(dose_gy))
        print(f"  {L:>18}: edep {edep:.6e} MeV | unc {unc*100:5.2f}% | "
              f"massa {vol*dens/1000.0:8.4f} g | {dose_gy*1e16:6.3f} x1e-16 Gy/hist")

    ratio = results["layers"]["basal"]["edep_MeV"] / results["layers"]["epidermis"]["edep_MeV"]
    results["edep_ratio_basal_over_epi"] = float(ratio)
    print(f"\nRASIO edep basal/epidermis = {ratio:.4f}")
    if 0.05 <= ratio <= 0.25:
        print("=> MEKANISME TERBATAS-VOLUME TERKONFIRMASI (Resep A sah)")
        results["mechanism"] = "volume_restricted"
    else:
        print("=> MEKANISME BOUNDING-BOX TERDETEKSI (Resep A TIDAK sah; pakai Resep B)")
        results["mechanism"] = "bbox_contaminated"

    prof = read_image(OUT / "cyl_dose.mhd").flatten()
    uncp = read_image(OUT / "cyl_unc.mhd").flatten()
    depths = [0.05, 0.7, 2.0, 3.0, 5.0]
    print("\n--- profil kedalaman (aktor silinder, bin 0.1 mm) ---")
    results["profile"] = {}
    for dp in depths:
        i = int(round((23.0 - dp) / 0.1 - 0.5))
        if 0 <= i < len(prof):
            print(f"  kedalaman {dp:4.2f} mm: dose {prof[i]:.6e} (unc {uncp[i]*100:5.2f}%)")
            results["profile"][f"{dp}mm"] = dict(dose=float(prof[i]), unc=float(uncp[i]))

    with open(OUT / "stage30_mechanism_test.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nRingkasan:", OUT / "stage30_mechanism_test.json")
    print("=" * 78)
    print("TAHAP 3.1 SELESAI")
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[TAHAP 3.1 GAGAL] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)