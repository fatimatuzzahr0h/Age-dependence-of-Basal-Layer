"""
Tahap 1.1 - Definisi Geometri Sumber Aplikator Sr-90 (Coelho 2011)
Geometri: active layer + substrate Ag + window PE 1.0 mm + barrier PE 10.0 mm
Verifikasi: simulasi 1 ms (~3700 peluruhan) + cek dose != 0
Cara jalankan: python 03_source_geometry_definition.py
"""

import sys
import numpy as np
import opengate as gate


def main():
    print("=" * 70)
    print("TAHAP 1.1 - DEFINISI GEOMETRI SUMBER APLIKATOR Sr-90")
    print("=" * 70)

    sim = gate.Simulation()
    sim.random_engine = "MersenneTwister"
    sim.random_seed = 123456789
    sim.number_of_threads = 1
    sim.visu = False
    sim.output_dir = "./source_geometry_output"

    mm = gate.g4_units.mm
    Bq = gate.g4_units.Bq
    second = gate.g4_units.second

    active_diameter = 22.57 * mm
    active_thickness = 0.01 * mm
    active_material = "G4_Sr"

    substrate_diameter = 22.57 * mm
    substrate_thickness = 0.1 * mm
    substrate_material = "G4_Ag"

    window_diameter = 24.0 * mm
    window_thickness = 1.0 * mm
    window_material = "G4_POLYETHYLENE"

    barrier_diameter = 26.0 * mm
    barrier_thickness = 10.0 * mm
    barrier_material = "G4_POLYETHYLENE"

    sim.world.size = [60 * mm, 60 * mm, 60 * mm]
    sim.world.material = "G4_AIR"

    active = sim.add_volume("Tubs", name="active_layer")
    active.rmin = 0
    active.rmax = active_diameter / 2
    active.dz = active_thickness / 2
    active.translation = [0, 0, -(window_thickness + substrate_thickness + active_thickness / 2)]
    active.material = active_material

    substrate = sim.add_volume("Tubs", name="substrate")
    substrate.rmin = 0
    substrate.rmax = substrate_diameter / 2
    substrate.dz = substrate_thickness / 2
    substrate.translation = [0, 0, -(window_thickness + substrate_thickness / 2)]
    substrate.material = substrate_material

    window = sim.add_volume("Tubs", name="window")
    window.rmin = 0
    window.rmax = window_diameter / 2
    window.dz = window_thickness / 2
    window.translation = [0, 0, -(window_thickness / 2)]
    window.material = window_material

    barrier = sim.add_volume("Tubs", name="primary_barrier")
    barrier.rmin = 0
    barrier.rmax = barrier_diameter / 2
    barrier.dz = barrier_thickness / 2
    barrier.translation = [0, 0, -(window_thickness + substrate_thickness + active_thickness + barrier_thickness / 2)]
    barrier.material = barrier_material

    source = sim.add_source("GenericSource", name="sr90_source")
    source.particle = "ion 38 90"
    source.position.type = "disc"
    source.position.radius = active_diameter / 2
    source.position.translation = [0, 0, active.translation[2]]
    source.direction.type = "iso"
    source.activity = 3.7e6 * Bq

    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option4"
    sim.physics_manager.enable_decay = True
    sim.physics_manager.set_production_cut("world", "all", 1 * mm)
    sim.physics_manager.set_production_cut("active_layer", "all", 0.01 * mm)
    sim.physics_manager.set_production_cut("substrate", "all", 0.01 * mm)
    sim.physics_manager.set_production_cut("window", "all", 0.01 * mm)
    sim.physics_manager.set_production_cut("primary_barrier", "all", 0.1 * mm)

    target = sim.add_volume("Box", name="verification_target")
    target.size = [30 * mm, 30 * mm, 3 * mm]
    target.translation = [0, 0, 1.5 * mm]
    target.material = "G4_POLYETHYLENE"

    dose_actor = sim.add_actor("DoseActor", name="verification_dose")
    dose_actor.attached_to = "verification_target"
    dose_actor.size = [1, 1, 1]
    dose_actor.spacing = [30 * mm, 30 * mm, 3 * mm]
    dose_actor.dose.active = True
    dose_actor.dose.output_filename = "verification_dose.mhd"

    print("")
    print("--- Verifikasi Dimensi Geometri (vs Coelho 2011) ---")
    print(f"1. Lapisan aktif : diameter {active_diameter / mm:.2f} mm, tebal {active_thickness / mm:.3f} mm (asumsi MC 10 um)")
    print(f"2. Substrate     : diameter {substrate_diameter / mm:.2f} mm, tebal {substrate_thickness / mm:.3f} mm, material {substrate_material}")
    print(f"3. Window        : diameter {window_diameter / mm:.2f} mm, tebal {window_thickness / mm:.2f} mm (paper: ~1.0 mm)")
    print(f"4. Barrier       : diameter {barrier_diameter / mm:.2f} mm, tebal {barrier_thickness / mm:.2f} mm (paper: ~10.0 mm)")

    print("")
    print("--- Volume Komponen ---")
    vol_active = np.pi * (active_diameter / 2) ** 2 * active_thickness
    vol_substrate = np.pi * (substrate_diameter / 2) ** 2 * substrate_thickness
    vol_window = np.pi * (window_diameter / 2) ** 2 * window_thickness
    vol_barrier = np.pi * (barrier_diameter / 2) ** 2 * barrier_thickness
    print(f"Volume aktif    : {vol_active / mm ** 3:.4f} mm3")
    print(f"Volume substrate: {vol_substrate / mm ** 3:.4f} mm3")
    print(f"Volume window   : {vol_window / mm ** 3:.4f} mm3")
    print(f"Volume barrier  : {vol_barrier / mm ** 3:.4f} mm3")

    print("")
    print("--- Verifikasi Sumber Sr-90 ---")
    print(f"Partikel : {source.particle}")
    print(f"Activity : {source.activity / Bq:.2e} Bq")
    print("Mode     : activity-driven (bypass bug GenericIon mode-N)")

    print("")
    print("--- Simulasi Verifikasi (1 ms, sekitar 3700 peluruhan) ---")
    sim.run_timing_intervals = [[0, 1e-3 * second]]

    try:
        sim.run(start_new_process=True)
        print("OK: Simulasi berhasil tanpa error geometri.")
    except Exception as e:
        print(f"ERROR saat simulasi: {e}")
        sys.exit(1)

    try:
        import itk
        from pathlib import Path
        dose_path = Path(sim.output_dir) / "verification_dose.mhd"
        if dose_path.exists():
            arr = itk.array_from_image(itk.imread(str(dose_path))).astype(float)
            total_dose = float(arr.sum())
            if total_dose > 0:
                print(f"OK: Dosis terdeposit = {total_dose:.6e} Gy")
                print("OK: Sumber mode-activity menghasilkan track (bukan zero dose).")
            else:
                print("ERROR: Dosis = 0, sumber tidak menghasilkan track.")
                sys.exit(1)
        else:
            print(f"ERROR: File dose tidak ditemukan: {dose_path}")
            sys.exit(1)
    except ImportError:
        print("PERINGATAN: paket itk tidak tersedia, lewati cek dose otomatis.")

    print("")
    print("=" * 70)
    print("TAHAP 1.1 SELESAI")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[TAHAP 1.1 GAGAL] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)