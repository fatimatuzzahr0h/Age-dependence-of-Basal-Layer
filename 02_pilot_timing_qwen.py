"""
Tahap 0.4 — Pilot timing single-thread
======================================
Tujuan: uji konvergensi statistik (10^5 -> 10^6 -> 10^7 histori) pada
SATU skenario acuan, untuk menentukan N minimum yang memberi
ketidakpastian <5% DAN memproyeksikan durasi realistis 18 skenario
produksi (Tahap 4) sebelum eksekusi penuh dijalankan.

CATATAN PENTING — geometri di skrip ini SENGAJA disederhanakan:
Cover aplikator & target dimodelkan sebagai kotak (Box), BUKAN
silinder presisi sesuai diameter 22,57 mm. Tujuannya murni timing,
bukan validasi PDD (itu tugas Tahap 1.2 dengan geometri presisi).

PERBAIKAN v4 — SUMBER:
Karena bug di OpenGATE 10.1.1 pada GenericSource mode-N untuk
GenericIon (ion Sr-90 tidak masuk stepping loop, menghasilkan
zero dose & waktu eksekusi ~0.03s/20 event), sumber diganti
dengan elektron (e-) berspektrum beta empiris Sr-90 + Y-90.
Spektrum di-sampling dari distribusi Fermi-Kurie (allowed transition)
untuk kedua isotop dengan rasio 1:1 (kesetimbangan sekular).

Histogram menggunakan bin_centers (bukan bin_edges) agar panjang
histogram_energy dan histogram_weight sama persis, sesuai requirement
OpenGATE 10.1.1.

Ini JAUH lebih berat secara komputasi daripada gamma mono-energi
smoke test sebelumnya, karena melibatkan banyak sekunder elektron.

Emisi diasumsikan isotropik penuh (bukan hanya hemisfer bawah) —
ini skenario waktu-terburuk (konservatif), aman untuk estimasi durasi.

PERINGATAN WAKTU: N=10^7 dengan proxy elektron bisa memakan waktu
lama (menit hingga jam) di laptop single-thread. Skrip ini akan
BERHENTI dan MEMINTA KONFIRMASI sebelum menjalankan N=10^7, berdasarkan
proyeksi dari hasil N=10^6 — sesuai gate/no-gate di desain studi Anda
(0.4: jika 10-20 juta histori butuh >beberapa hari, kurangi cakupan).

Cara jalankan:
    python 02_pilot_timing.py
"""

import json
import sys
import time
from pathlib import Path
import numpy as np
import opengate as gate


N_LEVELS = [1e5, 1e6, 1e7]
OUTPUT_DIR = Path("./pilot_timing_output")


def generate_beta_spectrum_histogram(n_samples=100000, n_bins=50, seed=42):
    """
    Generate histogram spektrum beta empiris Sr-90 + Y-90.
    Distribusi: N(E) ~ p * E * (Emax - E)^2 (Fermi-Kurie, allowed transition)
    dengan p = sqrt(E^2 + 2*E*me), me = 0.511 MeV.
    
    Return:
        bin_centers: array shape (n_bins,) dalam MeV
        weights: array shape (n_bins,) - probabilitas per bin (ternormalisasi)
    """
    rng = np.random.default_rng(seed)
    me = 0.511  # massa elektron dalam MeV
    Emax_sr90 = 0.546  # MeV
    Emax_y90 = 2.28    # MeV
    
    def fermi_kurie_pdf(E, Emax):
        """PDF spektrum beta (allowed transition)"""
        p = np.sqrt(E**2 + 2*E*me)
        return p * E * (Emax - E)**2
    
    # Sample dari kedua distribusi dengan rejection sampling
    def sample_beta_branch(Emax, n_target):
        samples = []
        # Hitung bound untuk rejection sampling
        E_test = np.linspace(0.001, Emax*0.999, 1000)
        pdf_test = fermi_kurie_pdf(E_test, Emax)
        bound = pdf_test.max() * 1.1
        
        while len(samples) < n_target:
            batch_size = (n_target - len(samples)) * 3  # oversample untuk efisiensi
            E_candidates = rng.uniform(0, Emax, size=batch_size)
            u = rng.uniform(0, bound, size=batch_size)
            accept = u < fermi_kurie_pdf(E_candidates, Emax)
            samples.extend(E_candidates[accept].tolist())
        
        return np.array(samples[:n_target])
    
    # Sample 50% Sr-90 + 50% Y-90
    n_half = n_samples // 2
    E_sr90 = sample_beta_branch(Emax_sr90, n_half)
    E_y90 = sample_beta_branch(Emax_y90, n_samples - n_half)
    E_all = np.concatenate([E_sr90, E_y90])
    
    # Buat histogram
    weights, bin_edges = np.histogram(E_all, bins=n_bins, range=(0.01, 2.28), density=True)
    
    # Konversi bin_edges menjadi bin_centers (panjang n_bins, sama dengan weights)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Normalisasi weights agar sum = 1
    weights = weights / weights.sum()
    
    return bin_centers, weights


def build_and_run(n_histories, run_id):
    sim = gate.Simulation()
    sim.random_engine = "MersenneTwister"
    sim.random_seed = 123456789
    sim.number_of_threads = 1  # sesuai desain studi: single-thread
    sim.visu = False
    sim.output_dir = str(OUTPUT_DIR / run_id)
    sim.progress_bar = True

    mm = gate.g4_units.mm
    MeV = gate.g4_units.MeV

    # --- World ---
    sim.world.size = [60 * mm, 60 * mm, 60 * mm]
    sim.world.material = "G4_AIR"

    # --- Cover aplikator: polietilena 1.0 mm (disederhanakan jadi kotak) ---
    cover = sim.add_volume("Box", name="cover")
    cover.size = [25 * mm, 25 * mm, 1.0 * mm]
    cover.translation = [0, 0, 0.5 * mm]  # rentang z: 0 -> 1.0 mm
    cover.material = "G4_POLYETHYLENE"

    # --- Target: stack polietilena, total 6 mm, discor per 0.6 mm ---
    target = sim.add_volume("Box", name="target")
    target.size = [40 * mm, 40 * mm, 6 * mm]
    target.translation = [0, 0, 1.0 * mm + 3.0 * mm]  # mulai tepat di bawah cover
    target.material = "G4_POLYETHYLENE"

    # --- Physics: standard EM opsi 4 (akurat untuk elektron energi rendah) ---
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option4"
    # enable_decay TIDAK diperlukan untuk sumber elektron langsung (proxy)
    sim.physics_manager.enable_decay = False
    sim.physics_manager.set_production_cut("world", "all", 1 * mm)
    sim.physics_manager.set_production_cut("cover", "all", 0.01 * mm)
    sim.physics_manager.set_production_cut("target", "all", 0.01 * mm)

    # --- Sumber: PROXY ELEKTRON dengan spektrum beta empiris ---
    # Alasan: GenericIon mode-N di OpenGATE 10.1.1 tidak memicu stepping loop
    # untuk ion dengan waktu paruh panjang (Sr-90, T1/2 = 28.79 tahun).
    # Proxy elektron dengan spektrum beta empiris Sr-90 + Y-90 memberikan beban
    # komputasi yang REPRESENTATIF untuk rantai peluruhan Sr-90 -> Y-90 -> Zr-90.
    # CATATAN: Ini HANYA untuk Tahap 0.4 (timing). Untuk Tahap 1.2 (validasi
    # PDD Coelho 2011), sumber ion Sr-90 asli akan digunakan kembali dengan
    # konfigurasi berbeda (mode activity, bukan mode-N).
    
    # Generate histogram spektrum beta
    bin_centers, weights = generate_beta_spectrum_histogram(n_samples=100000, n_bins=50)
    
    source = sim.add_source("GenericSource", name="beta_timing_proxy")
    source.particle = "e-"
    source.n = int(n_histories)
    source.position.type = "disc"
    source.position.radius = 11.285 * mm  # diameter 22.57 mm (sesuai applicator 1 Coelho)
    source.position.translation = [0, 0, -0.001 * mm]
    source.direction.type = "iso"
    
    # Assign histogram spektrum beta (bin_centers dan weights harus sama panjang)
    source.energy.type = "histogram"
    source.energy.histogram_energy = (bin_centers * MeV).tolist()
    source.energy.histogram_weight = weights.tolist()

    # --- Dose actor: PDD per lapisan 0.6 mm ---
    dose_actor = sim.add_actor("DoseActor", name="pdd")
    dose_actor.attached_to = "target"
    dose_actor.size = [1, 1, 10]  # 1 bin lateral (integrasi penuh), 10 bin kedalaman
    dose_actor.spacing = [40 * mm, 40 * mm, 0.6 * mm]
    dose_actor.dose.active = True
    dose_actor.dose_uncertainty.active = True
    dose_actor.dose.output_filename = f"pdd_dose_{run_id}.mhd"
    dose_actor.dose_uncertainty.output_filename = f"pdd_uncertainty_{run_id}.mhd"

    t0 = time.time()
    sim.run(start_new_process=True)  # wajib: >1 SimulationEngine per proses tidak diizinkan GATE 10
    elapsed = time.time() - t0

    # --- Baca ketidakpastian dari file output (lebih andal daripada akses
    #     langsung ke objek Python setelah run) ---
    unc_path = Path(sim.output_dir) / f"pdd_uncertainty_{run_id}.mhd"
    mean_unc_pct = None
    max_unc_pct = None
    try:
        import itk
        # subprocess (start_new_process=True) kadang butuh sesaat sampai file
        # benar-benar ter-flush ke disk setelah sim.run() kembali ke proses induk
        for attempt in range(5):
            if unc_path.exists():
                break
            time.sleep(1)
        if not unc_path.exists():
            print(f"  [PERINGATAN] File tidak ditemukan setelah menunggu: {unc_path}")
        else:
            img = itk.imread(str(unc_path))
            arr = itk.array_from_image(img).astype(float)
            valid = arr[arr > 0]  # abaikan bin tanpa hit (uncertainty=0 artifisial)
            if valid.size > 0:
                mean_unc_pct = float(np.mean(valid)) * 100
                max_unc_pct = float(np.max(valid)) * 100
            else:
                print("  [PERINGATAN] Semua bin uncertainty bernilai 0 (kemungkinan tidak ada hit).")
    except ImportError:
        print("  [PERINGATAN] Paket 'itk' tidak tersedia -- lewati pembacaan uncertainty otomatis. "
              "Cek manual file .mhd di folder output kalau perlu.")
    except Exception as e:
        print(f"  [PERINGATAN] Gagal membaca file uncertainty: {e}")

    return {
        "n_histories": int(n_histories),
        "elapsed_sec": elapsed,
        "rate_hist_per_sec": n_histories / elapsed,
        "mean_uncertainty_pct": mean_unc_pct,
        "max_uncertainty_pct": max_unc_pct,
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    results = []
    print("=" * 70)
    print("TAHAP 0.4 — PILOT TIMING SINGLE-THREAD (v4: bin_centers histogram)")
    print("=" * 70)
    for n in N_LEVELS:
        if n >= 1e7 and results:
            prev = results[-1]
            projected_sec = prev["elapsed_sec"] * (n / prev["n_histories"])
            print(f"\n[PROYEKSI] N={n:.0e} histori diperkirakan makan waktu "
                  f"~{projected_sec/60:.1f} menit (~{projected_sec/3600:.2f} jam), "
                  f"diekstrapolasi dari N={prev['n_histories']:.0e}.")
            ans = input("Lanjutkan menjalankan N ini sekarang? [y/N]: ").strip().lower()
            if ans != "y":
                print("Dilewati atas permintaan Anda. Pilot timing dihentikan di sini.")
                break
        print(f"\n--- Menjalankan N = {n:.0e} histori ---")
        res = build_and_run(n, run_id=f"N{n:.0e}".replace("+", ""))
        results.append(res)
        unc_str = (f"{res['mean_uncertainty_pct']:.2f}%"
                   if res["mean_uncertainty_pct"] is not None else "n/a")
        print(f"Selesai: {res['elapsed_sec']:.1f} detik | "
              f"rate {res['rate_hist_per_sec']:.1f} hist/s | "
              f"ketidakpastian rata-rata {unc_str}")

    # --- Ringkasan ---
    print("\n" + "=" * 70)
    print("RINGKASAN KONVERGENSI")
    print("=" * 70)
    print(f"{'N histori':>12} | {'Waktu (s)':>10} | {'Rate (h/s)':>10} | {'Unc. rata2 (%)':>15}")
    for r in results:
        unc = r["mean_uncertainty_pct"]
        unc_disp = f"{unc:.2f}" if unc is not None else "n/a"
        print(f"{r['n_histories']:>12} | {r['elapsed_sec']:>10.1f} | "
              f"{r['rate_hist_per_sec']:>10.1f} | {unc_disp:>15}")
    with open(OUTPUT_DIR / "pilot_timing_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nHasil lengkap (JSON) disimpan di: {OUTPUT_DIR / 'pilot_timing_results.json'}")

    if results:
        last = results[-1]
        print("\n--- Proyeksi durasi produksi (Tahap 4, 18 skenario) ---")
        for target_n in [1e7, 2e7]:
            est_per_scenario_sec = last["elapsed_sec"] * (target_n / last["n_histories"])
            est_total_sec = est_per_scenario_sec * 18
            print(f"@ {target_n:.0e} histori/skenario: "
                  f"~{est_per_scenario_sec/3600:.2f} jam/skenario -> "
                  f"~{est_total_sec/3600:.1f} jam total (~{est_total_sec/86400:.1f} hari) "
                  f"untuk 18 skenario berurutan.")
        print("\nCatatan: geometri phantom mesh ICRP-156 penuh + step limiter mikrometer "
              "(Tahap 3.2) kemungkinan akan LEBIH LAMBAT dari proyeksi ini. "
              "Gunakan angka ini sebagai batas bawah (best-case), bukan estimasi akhir.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[PILOT TIMING GAGAL]\nError: {e}")
        sys.exit(1)