"""
Tahap 3.2 - Pilot produksi skoring (Resep A terkonfirmasi)
Resep final:
- DoseActor 1x1x1 per lapisan (volume_restricted, Resep A).
- DoseActor silinder profil kedalaman (250 bin x 0.1 mm) menempel world,
  rotasi = normal lokal, untuk dermis operasional (L1) dan cross-check
  spongiosa (L2/N4).
- Konversi dosis: D[Gy/(Bq.s)] = 2 * D[Gy/hist] (kesetimbangan sekuler
  Sr-90/Y-90: 1 Bq -> 2 elektron beta/detik).
Pilot: MRCP_01M, 3 standoff, N = 2e6 hist per skenario (~30 menit total).
Output: D:\\Brachy\\stage3_output\\pilot_MRCP_01M.json
CLI: python 21_production_pilot.py
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
STANDOFFS = [0.0, 0.5, 1.0]
N_HIST = 2_000_000
LAYERS = ["epidermis", "basal", "cranium_cortical", "cranium_spongiosa"]
ELEMENTS = ["H", "C", "N", "O", "Na", "Mg", "P", "S", "Cl", "K", "Ca", "Fe", "I"]
MAT_KEY = {"epidermis": "skin", "basal": "skin",
           "cranium_cortical": "cortical", "cranium_spongiosa": "spongiosa"}
MEV_TO_J = 1.602176634e-13
# Kesetimbangan sekuler Sr-90/Y-90: 1 Bq -> 2 elektron beta per detik
BQ_S_PER_HIST = 2.0


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


def run_scenario(tag, s, meds_sel, off, geo, a, n, h, R, out_dir):
    sim = gate.Simulation()
    sim.number_of_threads = 1
    sim.visu = False
    sim.progress_bar = True
    sim.output_dir = str(out_dir)
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
            made[mk] = (mname, m)
        v = sim.add_volume("TesselatedVolume", name=f"{tag}_{L}")
        v.file_name = str(WORK / tag / "stl" / f"layer_{L}.stl")
        v.material = made[mk][0]
        v.translation = [0 * mm, 0 * mm, 0 * mm]

    stack = [("window", 0.5, 0.5, 12.0, "G4_POLYETHYLENE", 0.01),
             ("substrate", 1.05, 0.05, 11.285, "G4_Ag", 0.01),
             ("active", 1.105, 0.005, 11.285, "G4_Sr", 0.01),
             ("barrier", 6.11, 5.0, 13.0, "G4_POLYETHYLENE", 0.1)]
    for name, tz, dz, rmax, mat, cut in stack:
        v = sim.add_volume("Tubs", name=f"{tag}_{name}")
        v.rmin = 0
        v.rmax = rmax * mm
        v.dz = dz * mm
        t = a + n * (s + tz) - n * ((a @ n) - h)
        v.translation = [t[0] * mm, t[1] * mm, t[2] * mm]
        v.rotation = R
        sim.physics_manager.set_production_cut(f"{tag}_{name}", "all", cut * mm)

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
        sim.physics_manager.set_production_cut(f"{tag}_{L}", "all", 0.01 * mm)

    # DoseActor per lapisan (Resep A - volume_restricted)
    for L in LAYERS:
        d = sim.add_actor("DoseActor", f"dose_{L}")
        d.attached_to = f"{tag}_{L}"
        d.size = [1, 1, 1]
        e = geo[L]["extent"]
        c = geo[L]["center"]
        d.spacing = [max(e[0], 0.01) * mm, max(e[1], 0.01) * mm, max(e[2], 0.01) * mm]
        d.translation = [c[0] * mm, c[1] * mm, c[2] * mm]
        d.edep.active = True
        d.dose.active = True
        d.dose_uncertainty.active = True
        d.edep.output_filename = f"edep_{L}_s{s:.1f}.mhd"
        d.dose.output_filename = f"dose_{L}_s{s:.1f}.mhd"
        d.dose_uncertainty.output_filename = f"unc_{L}_s{s:.1f}.mhd"

    # Aktor silinder profil kedalaman (dermis operasional L1 + cross-check)
    cyl = sim.add_actor("DoseActor", f"cyl_s{s:.1f}")
    cyl.attached_to = "world"
    cyl.size = [1, 1, 250]
    cyl.spacing = [40 * mm, 40 * mm, 0.1 * mm]
    ccenter = a + n * (-10.5) - n * ((a @ n) - h)
    cyl.translation = [ccenter[0] * mm, ccenter[1] * mm, ccenter[2] * mm]
    cyl.rotation = R
    cyl.edep.active = True
    cyl.dose.active = True
    cyl.dose_uncertainty.active = True
    cyl.edep.output_filename = f"cyl_edep_s{s:.1f}.mhd"
    cyl.dose.output_filename = f"cyl_dose_s{s:.1f}.mhd"
    cyl.dose_uncertainty.output_filename = f"cyl_unc_s{s:.1f}.mhd"

    t0 = time.time()
    sim.run(start_new_process=True)
    dt = time.time() - t0
    return dt


def main():
    OUT.mkdir(exist_ok=True)
    print("=" * 78)
    print(f"TAHAP 3.2 - PILOT PRODUKSI | {TAG} | N={N_HIST:.0e} | 3 standoff")
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

    meds_sel = pick_media(parse_media(TAG))
    results = []

    for s in STANDOFFS:
        out_dir = OUT / f"{TAG}_s{s:.1f}"
        out_dir.mkdir(exist_ok=True)
        print(f"\n--- standoff {s:.1f} mm ---")
        dt = run_scenario(TAG, s, meds_sel, off, geo, a, n, h, R, out_dir)
        print(f"run selesai {dt:.1f} s")

        row = dict(tag=TAG, standoff_mm=s, n_hist=N_HIST, layers={}, profile={})
        for L in LAYERS:
            dose_arr = read_image(out_dir / f"dose_{L}_s{s:.1f}.mhd")
            unc_arr = read_image(out_dir / f"unc_{L}_s{s:.1f}.mhd")
            dose_gy_hist = float(dose_arr.mean())
            unc_rel = float(unc_arr.flatten()[0])
            vol = summ[TAG][L]["volume_mm3"]
            dens = meds_sel[MAT_KEY[L]]["density"]
            mass_g = vol * dens / 1000.0
            dose_gy_bqs = dose_gy_hist * BQ_S_PER_HIST
            row["layers"][L] = dict(
                dose_Gy_per_hist=dose_gy_hist,
                dose_Gy_per_Bq_s=dose_gy_bqs,
                unc_rel=unc_rel,
                mass_g=mass_g
            )
            print(f"  {L:>18}: {dose_gy_hist:.3e} Gy/hist | "
                  f"{dose_gy_bqs:.3e} Gy/(Bq.s) | unc {unc_rel*100:.2f}%")

        # Profil kedalaman (dermis operasional L1)
        cyl_dose = read_image(out_dir / f"cyl_dose_s{s:.1f}.mhd").flatten()
        cyl_unc = read_image(out_dir / f"cyl_unc_s{s:.1f}.mhd").flatten()
        depths = [0.05, 0.1, 0.5, 0.7, 1.0, 2.0, 3.0, 5.0]
        for dp in depths:
            i = int(round((23.0 - dp) / 0.1 - 0.5))
            if 0 <= i < len(cyl_dose):
                row["profile"][f"{dp}mm"] = dict(
                    dose_Gy_per_hist=float(cyl_dose[i]),
                    dose_Gy_per_Bq_s=float(cyl_dose[i] * BQ_S_PER_HIST),
                    unc_rel=float(cyl_unc[i])
                )

        results.append(row)

    out_json = OUT / f"pilot_{TAG}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRingkasan pilot: {out_json}")
    print("=" * 78)
    print("TAHAP 3.2 PILOT SELESAI")
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[TAHAP 3.2 GAGAL] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)