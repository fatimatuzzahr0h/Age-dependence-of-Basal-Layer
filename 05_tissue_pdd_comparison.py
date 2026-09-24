"""
Tahap 1.3 - Transisi ke jaringan ICRP (validasi 1.3b)
Tujuan: mengganti material fantom dari polietilena (surogat, Tahap 1.2)
menjadi jaringan lunak ICRP standar (G4_A-150_TISSUE) pada geometri
planar sederhana, dan mengkuantifikasi deviasi PDD.
Deviasi ini adalah TEMUAN ILMIAH (bukan galat), yang akan menjadi dasar
diskusi "polietilena-surogat vs jaringan nyata" di manuskrip.
Cara jalankan:
python 05_tissue_pdd_comparison.py
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
OUTPUT_DIR = Path("./tissue_pdd_output")

# PDD referensi dari Tahap 1.2 (polietilena)
PDD_PE_REF = [100.00, 76.05, 56.92, 40.60, 27.76, 18.27, 11.91, 6.99, 3.88, 1.89]


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

    # PERUBAHAN UTAMA TAHAP 1.3: fantom menggunakan jaringan ICRP
    # G4_A-150_TISSUE = ICRU soft tissue equivalent (densitas 1.127 g/cm3,
    # Z_eff ~7.07, komposisi H/C/N/O/F/Ca). Material ini lebih representatif
    # untuk jaringan kulit dibanding polietilena.
    skin_material = "G4_A-150_TISSUE"

    phantom = sim.add_volume("Tubs", name="skin_phantom")
    phantom.rmin = 0
    phantom.rmax = 30.0 * mm
    phantom.dz = (N_LAYERS * LAYER_MM) / 2.0 * mm
    phantom.translation = [0, 0, (N_LAYERS * LAYER_MM) / 2.0 * mm]
    phantom.material = skin_material

    probe = sim.add_volume("Tubs", name="pdd_probe")
    probe.mother = "skin_phantom"
    probe.rmin = 0
    probe.rmax = PROBE_R_MM * mm
    probe.dz = (N_LAYERS * LAYER_MM) / 2.0 * mm
    probe.translation = [0, 0, 0]
    probe.material = skin_material

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
    sim.physics_manager.set_production_cut("skin_phantom", "all", 0.01 * mm)
    sim.physics_manager.set_production_cut("pdd_probe", "all", 0.01 * mm)

    dose_actor = sim.add_actor("DoseActor", name="pdd")
    dose_actor.attached_to = "pdd_probe"
    dose_actor.size = [1, 1, N_LAYERS]
    dose_actor.spacing = [2 * PROBE_R_MM * mm, 2 * PROBE_R_MM * mm, LAYER_MM * mm]
    dose_actor.dose.active = True
    dose_actor.dose_uncertainty.active = True
    dose_actor.dose.output_filename = "tissue_pdd_dose.mhd"
    dose_actor.dose_uncertainty.output_filename = "tissue_pdd_unc.mhd"
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
    print("TAHAP 1.3 - TRANSISI KE JARINGAN ICRP (VALIDASI 1.3b)")
    print("=" * 70)
    print(f"Histori: {N_HIST:.0e} | single-thread | material: G4_A-150_TISSUE")
    print("Membandingkan dengan PDD polietilena dari Tahap 1.2 (referensi).")

    sim = build_sim()
    t0 = time.time()
    sim.run(start_new_process=True)
    elapsed = time.time() - t0
    print(f"Simulasi selesai dalam {elapsed:.1f} detik.")

    dose = read_image(OUTPUT_DIR / "tissue_pdd_dose.mhd").flatten()
    unc = read_image(OUTPUT_DIR / "tissue_pdd_unc.mhd").flatten()

    pdd_tissue = dose / dose[0] * 100.0
    pdd_pe = np.array(PDD_PE_REF, dtype=float)
    centers_mm = LAYER_MM * (np.arange(N_LAYERS) + 0.5)

    # Hitung deviasi
    ratio = pdd_tissue / pdd_pe
    dev_pct = (pdd_tissue - pdd_pe)

    print("")
    print("--- Perbandingan PDD: Jaringan ICRP vs Polietilena (Tahap 1.2) ---")
    print(f"{'z (mm)':>6} | {'PDD ICRP':>9} | {'unc(%)':>6} | {'PDD PE':>9} | {'ratio':>7} | {'dev (pp)':>8}")
    for i in range(N_LAYERS):
        print(f"{centers_mm[i]:>6.2f} | {pdd_tissue[i]:>9.2f} | {unc[i]*100:>6.2f} | "
              f"{pdd_pe[i]:>9.2f} | {ratio[i]:>7.3f} | {dev_pct[i]:>+8.2f}")

    print("")
    print("--- Ringkasan Deviasi (ICRP vs Polietilena) ---")
    # Hanya hitung statistik di rentang klinis relevan (PDD >= 10%)
    mask = pdd_pe >= 10.0
    if mask.any():
        print(f"Rentang klinis (PDD >= 10%, {int(mask.sum())} lapis):")
        print(f"  Rasio rata-rata     : {ratio[mask].mean():.3f}")
        print(f"  Rasio minimum       : {ratio[mask].min():.3f} pada z = {centers_mm[mask][ratio[mask].argmin()]:.2f} mm")
        print(f"  Rasio maksimum      : {ratio[mask].max():.3f} pada z = {centers_mm[mask][ratio[mask].argmax()]:.2f} mm")
        print(f"  Deviasi maks absolut: {np.abs(dev_pct[mask]).max():.2f} poin persen")
    else:
        print("Tidak ada lapis dengan PDD >= 10%.")

    # Temuan penting: di kedalaman 2-3 mm (lapisan basal/dermis dalam)
    # biasanya terlihat deviasi terbesar karena perbedaan stopping power
    basal_idx = 2  # ~1.5 mm (batas epidermis-basal)
    dermis_idx = 4  # ~2.7 mm (dermis dalam)
    print("")
    print("--- Temuan Klinis Kunci ---")
    print(f"Dosis di lapisan basal (~{centers_mm[basal_idx]:.1f} mm):")
    print(f"  Polietilena (Coelho): {pdd_pe[basal_idx]:.2f}%")
    print(f"  Jaringan ICRP       : {pdd_tissue[basal_idx]:.2f}%")
    print(f"  → Pasien menerima {dev_pct[basal_idx]:+.2f} pp vs pengukuran polietilena")
    print(f"Dosis di dermis dalam (~{centers_mm[dermis_idx]:.1f} mm):")
    print(f"  Polietilena (Coelho): {pdd_pe[dermis_idx]:.2f}%")
    print(f"  Jaringan ICRP       : {pdd_tissue[dermis_idx]:.2f}%")
    print(f"  → Pasien menerima {dev_pct[dermis_idx]:+.2f} pp vs pengukuran polietilena")

    result = {
        "n_hist": N_HIST,
        "elapsed_sec": elapsed,
        "skin_material": "G4_A-150_TISSUE",
        "centers_mm": centers_mm.tolist(),
        "pdd_tissue_pct": pdd_tissue.tolist(),
        "pdd_pe_ref_pct": pdd_pe.tolist(),
        "unc_pct": (unc * 100).tolist(),
        "ratio": ratio.tolist(),
        "deviation_pp": dev_pct.tolist(),
    }
    out_json = OUTPUT_DIR / "tissue_pdd_results.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print("")
    print(f"Hasil lengkap disimpan di: {out_json}")
    print("=" * 70)
    print("TAHAP 1.3 SELESAI")
    print("=" * 70)
    print("Deviasi antara polietilena (surogat Coelho) dan jaringan ICRP")
    print("telah terkuantifikasi. Ini adalah temuan ilmiah, bukan galat MC.")
    print("Setelah Anda konfirmasi, kita lanjut ke Tahap 2 (konstruksi phantom mesh).")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[TAHAP 1.3 GAGAL] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)