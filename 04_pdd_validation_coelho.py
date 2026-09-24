"""
Tahap 1.2 - Validasi PDD terhadap Coelho 2011 (Applicator 1)
Geometri sumber: identik Tahap 1.1 (stack Tubs presisi, diameter 22.57 mm).
Fantom kulit: silinder polietilena (Tubs rmin=0), 10 lapis x 0.6 mm (0-6 mm),
sesuai plate Coelho 2011 Section 3.1. Probe Tubs diameter 5 mm di sumbu
tengah meniru volume sensitif mini-extrapolation chamber.
Sumber: elektron dengan direct spectrum sampling (Fermi-Kurie dua cabang:
Sr-90 Emax 0.546 MeV dan Y-90 Emax 2.28 MeV, bobot kesetimbangan 0.5:0.5),
menggunakan tipe energi "histogram" (bin centers + weights, panjang sama),
kombinasi yang sudah terbukti berfungsi di pilot timing Tahap 0.4.
Validasi: PDD MC vs exp(a+bz+cz2+dz3+ez4+fz5) koefisien Tabel 1 Coelho
(Applicator 1). Konvensi satuan z (mm vs cm) diuji keduanya karena paper
tidak menyatakannya; konvensi dengan selisih terkecil diadopsi dan dicatat.
Cara jalankan:
python 04_pdd_validation_coelho.py
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import opengate as gate

N_HIST = 10_000_000
LAYER_MM = 0.6
N_LAYERS = 10
PROBE_R_MM = 2.5
OUTPUT_DIR = Path("./pdd_validation_output")

COELHO_A1 = dict(a=0.5542, b=-0.5931, c=0.06049, d=-0.0269, e=0.00358, f=-0.0002408)


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


def coelho_curve(z_mm, unit):
    z = np.asarray(z_mm, dtype=float)
    if unit == "cm":
        z = z / 10.0
    c = COELHO_A1
    poly = (c["a"] + c["b"] * z + c["c"] * z ** 2 + c["d"] * z ** 3
            + c["e"] * z ** 4 + c["f"] * z ** 5)
    return np.exp(poly)


def build_sim():
    sim = gate.Simulation()
    sim.random_engine = "MersenneTwister"
    sim.random_seed = 123456789
    sim.number_of_threads = 1
    sim.visu = False
    sim.output_dir = str(OUTPUT_DIR)
    sim.progress_bar = True

    mm = gate.g4_units.mm
    MeV = gate.g4_units.MeV

    sim.world.size = [60 * mm, 60 * mm, 60 * mm]
    sim.world.material = "G4_AIR"

    active_diameter = 22.57 * mm
    active_thickness = 0.01 * mm
    substrate_thickness = 0.1 * mm
    window_thickness = 1.0 * mm
    barrier_thickness = 10.0 * mm

    active = sim.add_volume("Tubs", name="active_layer")
    active.rmin = 0
    active.rmax = active_diameter / 2
    active.dz = active_thickness / 2
    active.translation = [0, 0, -(window_thickness + substrate_thickness + active_thickness / 2)]
    active.material = "G4_Sr"

    substrate = sim.add_volume("Tubs", name="substrate")
    substrate.rmin = 0
    substrate.rmax = active_diameter / 2
    substrate.dz = substrate_thickness / 2
    substrate.translation = [0, 0, -(window_thickness + substrate_thickness / 2)]
    substrate.material = "G4_Ag"

    window = sim.add_volume("Tubs", name="window")
    window.rmin = 0
    window.rmax = 24.0 * mm / 2
    window.dz = window_thickness / 2
    window.translation = [0, 0, -(window_thickness / 2)]
    window.material = "G4_POLYETHYLENE"

    barrier = sim.add_volume("Tubs", name="primary_barrier")
    barrier.rmin = 0
    barrier.rmax = 26.0 * mm / 2
    barrier.dz = barrier_thickness / 2
    barrier.translation = [0, 0, -(window_thickness + substrate_thickness + active_thickness + barrier_thickness / 2)]
    barrier.material = "G4_POLYETHYLENE"

    phantom = sim.add_volume("Tubs", name="pe_phantom")
    phantom.rmin = 0
    phantom.rmax = 30.0 * mm
    phantom.dz = (N_LAYERS * LAYER_MM) / 2.0 * mm
    phantom.translation = [0, 0, (N_LAYERS * LAYER_MM) / 2.0 * mm]
    phantom.material = "G4_POLYETHYLENE"

    probe = sim.add_volume("Tubs", name="pdd_probe")
    probe.mother = "pe_phantom"
    probe.rmin = 0
    probe.rmax = PROBE_R_MM * mm
    probe.dz = (N_LAYERS * LAYER_MM) / 2.0 * mm
    probe.translation = [0, 0, 0]
    probe.material = "G4_POLYETHYLENE"

    source = sim.add_source("GenericSource", name="sr90_beta_source")
    source.particle = "e-"
    centers, weights = fermi_kurie_spectrum()
    source.energy.type = "histogram"
    source.energy.histogram_energy = (centers * MeV).tolist()
    source.energy.histogram_weight = weights.tolist()
    source.n = N_HIST
    source.position.type = "disc"
    source.position.radius = active_diameter / 2
    source.position.translation = [0, 0, active.translation[2]]
    source.direction.type = "iso"

    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option4"
    sim.physics_manager.enable_decay = False
    sim.physics_manager.set_production_cut("world", "all", 1 * mm)
    sim.physics_manager.set_production_cut("active_layer", "all", 0.01 * mm)
    sim.physics_manager.set_production_cut("substrate", "all", 0.01 * mm)
    sim.physics_manager.set_production_cut("window", "all", 0.01 * mm)
    sim.physics_manager.set_production_cut("primary_barrier", "all", 0.1 * mm)
    sim.physics_manager.set_production_cut("pe_phantom", "all", 0.01 * mm)
    sim.physics_manager.set_production_cut("pdd_probe", "all", 0.01 * mm)

    dose_actor = sim.add_actor("DoseActor", name="pdd")
    dose_actor.attached_to = "pdd_probe"
    dose_actor.size = [1, 1, N_LAYERS]
    dose_actor.spacing = [2 * PROBE_R_MM * mm, 2 * PROBE_R_MM * mm, LAYER_MM * mm]
    dose_actor.dose.active = True
    dose_actor.dose_uncertainty.active = True
    dose_actor.dose.output_filename = "pdd_dose.mhd"
    dose_actor.dose_uncertainty.output_filename = "pdd_unc.mhd"
    return sim


def read_image(path, retries=5):
    import itk
    for _ in range(retries):
        if Path(path).exists():
            break
        time.sleep(1)
    img = itk.imread(str(path))
    return itk.array_from_image(img).astype(float)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("=" * 70)
    print("TAHAP 1.2 - VALIDASI PDD TERHADAP COELHO 2011 (APPLICATOR 1)")
    print("=" * 70)
    print(f"Histori: {N_HIST:.0e} | single-thread | spektrum Fermi-Kurie Sr/Y")

    sim = build_sim()
    t0 = time.time()
    sim.run(start_new_process=True)
    elapsed = time.time() - t0
    print(f"Simulasi selesai dalam {elapsed:.1f} detik.")

    dose = read_image(OUTPUT_DIR / "pdd_dose.mhd").flatten()
    unc = read_image(OUTPUT_DIR / "pdd_unc.mhd").flatten()

    pdd = dose / dose[0] * 100.0
    centers_mm = LAYER_MM * (np.arange(N_LAYERS) + 0.5)

    curves = {}
    for unit in ("mm", "cm"):
        f = coelho_curve(centers_mm, unit)
        f = f / f[0] * 100.0
        curves[unit] = f

    mask = pdd >= 20.0
    diffs = {u: np.abs(pdd[mask] - curves[u][mask]) for u in ("mm", "cm")}
    best = min(diffs, key=lambda u: diffs[u].max())

    print("")
    print("--- Tabel PDD (probe sumbu tengah, normalisasi lapis pertama) ---")
    print(f"{'z pusat (mm)':>12} | {'PDD MC (%)':>10} | {'unc (%)':>8} | "
          f"{'Coelho mm':>10} | {'Coelho cm':>10}")
    for i in range(N_LAYERS):
        print(f"{centers_mm[i]:>12.2f} | {pdd[i]:>10.2f} | {unc[i]*100:>8.2f} | "
              f"{curves['mm'][i]:>10.2f} | {curves['cm'][i]:>10.2f}")

    print("")
    print(f"Konvensi satuan z diadopsi (selisih terkecil): {best}")
    print(f"Selisih maksimum pada rentang PDD >= 20%: {diffs[best].max():.2f} poin persen")
    print(f"Selisih rata-rata pada rentang tersebut  : {diffs[best].mean():.2f} poin persen")

    fit_mask = pdd >= 10.0
    deg = min(5, int(fit_mask.sum()) - 1)
    coef = np.polyfit(centers_mm[fit_mask], np.log(pdd[fit_mask] / 100.0), deg)
    print("")
    print(f"Fit polinomial derajat-{deg} dari kurva MC (z dalam mm), ln(PDD):")
    names = ["a", "b", "c", "d", "e", "f"]
    pad = [np.nan] * (6 - len(coef))
    vals = list(coef) + pad
    for nm, v in zip(names, vals):
        vs = f"{v: .5f}" if v == v else "   n/a"
        print(f"  {nm} = {vs}")
    print("Koefisien Coelho Tabel 1 (Applicator 1) untuk rujukan:")
    for nm in names:
        print(f"  {nm} = {COELHO_A1[nm]: .5f}")

    result = {
        "n_hist": N_HIST,
        "elapsed_sec": elapsed,
        "centers_mm": centers_mm.tolist(),
        "pdd_mc_pct": pdd.tolist(),
        "unc_pct": (unc * 100).tolist(),
        "coelho_mm_pct": curves["mm"].tolist(),
        "coelho_cm_pct": curves["cm"].tolist(),
        "adopted_unit": best,
        "max_diff_pct_points": float(diffs[best].max()),
        "mean_diff_pct_points": float(diffs[best].mean()),
        "fit_coeffs_mm": [float(v) for v in coef],
    }
    out_json = OUTPUT_DIR / "pdd_validation_results.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print("")
    print(f"Hasil lengkap disimpan di: {out_json}")
    print("=" * 70)
    print("TAHAP 1.2 SELESAI")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[TAHAP 1.2 GAGAL] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)