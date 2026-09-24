"""
Smoke test — verifikasi instalasi GATE 10 (OpenGATE / Geant4 11.4.2)
======================================================================
Tujuan: memastikan opengate_core, physics engine, sumber, dan dose
actor berfungsi normal di mesin ini — TANPA bergantung pada data test
eksternal (gitlab.in2p3.fr) yang sempat timeout.

Geometri: kotak air sederhana disinari gamma monoenergetik dari satu
sisi. Bukan validasi fisika (itu Tahap 1.2), hanya cek instalasi.

Cara jalankan:
    python smoke_test_gate10.py
"""

import time
import sys

import opengate as gate


def main():
    print("=" * 60)
    print("GATE 10 SMOKE TEST")
    print("=" * 60)

    sim = gate.Simulation()

    # --- Pengaturan umum ---
    sim.random_engine = "MersenneTwister"
    sim.random_seed = 123456789          # fixed seed -> hasil reproducible
    sim.number_of_threads = 1            # single-thread, sesuai desain studi
    sim.visu = False                     # Qt tidak tersedia di build Windows ini
    sim.output_dir = "./smoke_test_output"
    sim.progress_bar = True

    # --- Unit ---
    mm = gate.g4_units.mm
    MeV = gate.g4_units.MeV

    # --- World ---
    sim.world.size = [40 * mm, 40 * mm, 40 * mm]
    sim.world.material = "G4_AIR"

    # --- Geometri: kotak air (surogat jaringan) ---
    phantom = sim.add_volume("Box", name="phantom")
    phantom.size = [20 * mm, 20 * mm, 20 * mm]
    phantom.translation = [0, 0, 0]
    phantom.material = "G4_WATER"

    # --- Fisika ---
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option3"
    sim.physics_manager.set_production_cut("world", "all", 1 * mm)
    sim.physics_manager.set_production_cut("phantom", "all", 0.01 * mm)

    # --- Sumber: gamma monoenergetik 0.662 MeV (setara Cs-137), collimated ---
    source = sim.add_source("GenericSource", name="point_source")
    source.particle = "gamma"
    source.n = 10000  # jumlah histori primer untuk smoke test (cepat)
    source.position.type = "point"
    source.position.translation = [0, 0, -15 * mm]
    source.direction.type = "momentum"
    source.direction.momentum = [0, 0, 1]
    source.energy.type = "mono"
    source.energy.mono = 0.662 * MeV

    # --- Dose actor ---
    dose_actor = sim.add_actor("DoseActor", name="dose")
    dose_actor.attached_to = "phantom"
    dose_actor.size = [20, 20, 20]
    dose_actor.spacing = [1 * mm, 1 * mm, 1 * mm]
    dose_actor.dose.active = True
    dose_actor.dose_uncertainty.active = True
    dose_actor.edep.output_filename = "smoke_test_edep.mhd"
    dose_actor.dose.output_filename = "smoke_test_dose.mhd"

    # --- Jalankan ---
    print("\nMenjalankan simulasi (10.000 histori, single-thread)...\n")
    t0 = time.time()
    sim.run()
    elapsed = time.time() - t0

    # --- Ringkasan ---
    print("\n" + "=" * 60)
    print("SMOKE TEST SELESAI")
    print("=" * 60)
    n_events = int(sum(source.n)) if hasattr(source.n, "__len__") else int(source.n)
    print(f"Waktu simulasi total     : {elapsed:.2f} detik")
    print(f"Histori dijalankan       : {n_events}")
    print(f"Rate (histori/detik)     : {n_events / elapsed:.1f}")
    print(f"Output disimpan di       : {sim.output_dir}")
    print("Status                   : GATE 10 / Geant4 berfungsi normal.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[SMOKE TEST GAGAL]")
        print(f"Error: {e}")
        sys.exit(1)