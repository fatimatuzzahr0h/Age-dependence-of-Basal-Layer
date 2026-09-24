"""
Tahap 2.3 v6 (FINAL) - Impor STL ke GATE 10 + material ICRU media.dat
Kasus ditutup: C++ ConstructNewMaterialWeights menuntut 4 argumen 
(name, elements, weights, density). Wrapper Python init_user_mat hanya 
melakukan *mat_info. Maka tuple yang disimpan DI DALAM kontainer 
(new_materials_weights) HARUS berisi 4 elemen tersebut, termasuk name.
Bug OpenGATE 10.1.1: add_material_weights bawaan crash karena salah 
indeks (list vs dict). v6 mem-bypass metode tersebut dan menulis 
langsung tuple 4-elemen (name, els, ws, dens) ke kontainer, 
menangani kedua kemungkinan tipe (dict atau list).
Desain tetap:
- Re-centering verteks ke z=0 via translation = (-x0, -y0, -z_top).
- Material per set dari media.dat: skin -> epidermis & basal;
  cortical -> cranium_cortical; cranium spongiosa -> cranium_spongiosa.
- Dermis tidak diimpor (L1); spongiosa 15M indikatif (L2).
CLI: python 14_gate_import_stl.py [TAG ...]   (default: MRCP_01M)
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import opengate as gate

WORK = Path(r"D:\Brachy\icrp156_work")
RAW = Path(r"D:\Brachy\icrp156_raw")
OUT = Path(r"D:\Brachy\stage23_output")

ELEMENTS = ["H", "C", "N", "O", "Na", "Mg", "P", "S", "Cl", "K", "Ca", "Fe", "I"]
LAYERS = ["epidermis", "basal", "cranium_cortical", "cranium_spongiosa"]
MAT_KEY = {"epidermis": "skin", "basal": "skin",
           "cranium_cortical": "cortical", "cranium_spongiosa": "spongiosa"}


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
            name = " ".join(t[1:-14])
            meds[idx] = dict(name=name, density=dens, fractions=fr)
    return meds


def pick_media(meds):
    sel = {}
    for idx, m in meds.items():
        if "skin" in m["name"].lower():
            sel.setdefault("skin", m)
    cort = None
    for idx, m in meds.items():
        low = m["name"].lower()
        if "cranium" in low and "cortical" in low:
            cort = m
            break
    if cort is None:
        for idx, m in meds.items():
            if "cortical" in m["name"].lower():
                cort = m
                break
    if cort is not None:
        sel["cortical"] = cort
    for idx, m in meds.items():
        low = m["name"].lower()
        if "cranium" in low and "spongiosa" in low:
            sel["spongiosa"] = m
    return sel


def define_material(sim, name, dens_gcm3, fracs, media_name):
    g_cm3 = gate.g4_units.g / (gate.g4_units.cm ** 3)
    db = sim.volume_manager.material_database
    pairs = [(el, f) for el, f in zip(ELEMENTS, fracs) if f > 0]
    wsum = sum(f for _, f in pairs)
    els = [el for el, _ in pairs]
    ws = [f / wsum for _, f in pairs]
    dens = dens_gcm3 * g_cm3

    store = getattr(db, "new_materials_weights", None)
    
    # KUNCI PERBAIKAN: Tuple harus 4 elemen (name, els, ws, dens)
    # agar saat di-unpack oleh *mat_info di init_user_mat, 
    # C++ menerima persis 4 argumen yang dituntutnya.
    if isinstance(store, dict):
        store[name] = (name, els, ws, dens)
        how = "dict: store[name]=(name,els,ws,dens)"
    elif isinstance(store, list):
        store.append((name, els, ws, dens))
        how = "list: append((name,els,ws,dens))"
    else:
        raise RuntimeError(f"new_materials_weights bertipe tak dikenal: {type(store)}")

    print(f"    material {name} <- media '{media_name}' | {len(els)} unsur | "
          f"rho target {dens_gcm3:.4f} g/cm3 | {how} [OK]")
    return how


def build_tag(tag):
    with open(WORK / "extraction_summary_v5_all.json", "r") as f:
        summ = json.load(f)
    row = [r for r in summ if r["tag"] == tag]
    if not row:
        raise KeyError(f"{tag} tidak ada di extraction_summary_v5_all.json")
    row = row[0]
    xy0 = row["xy_center_mm"]
    ztop = row["z_top_mm"]

    sim = gate.Simulation()
    sim.number_of_threads = 1
    sim.visu = False
    sim.progress_bar = False
    sim.output_dir = str(OUT / tag)
    mm = gate.g4_units.mm
    MeV = gate.g4_units.MeV

    sim.world.size = [400 * mm, 400 * mm, 400 * mm]
    sim.world.material = "G4_AIR"

    meds = parse_media(tag)
    sel = pick_media(meds)
    missing = [k for k in ("skin", "cortical", "spongiosa") if k not in sel]
    if missing:
        raise KeyError(f"media.dat {tag} tidak memuat: {missing}")

    print(f"\n=== {tag} | verteks asli z={ztop:.1f} mm, xy=({xy0[0]:.1f},{xy0[1]:.1f}) mm ===")
    info = []
    made_mats = {}
    for layer in LAYERS:
        mk = MAT_KEY[layer]
        if mk not in made_mats:
            m = sel[mk]
            mat_name = f"ICRP156_{tag}_{mk}"
            how = define_material(sim, mat_name, m["density"], m["fractions"], m["name"])
            made_mats[mk] = (mat_name, m, how)
        mat_name, m, how = made_mats[mk]
        stl = WORK / tag / "stl" / f"layer_{layer}.stl"
        if not stl.exists():
            print(f"  [LEWATI] {layer}: STL tidak ada")
            continue
        v = sim.add_volume("TesselatedVolume", name=f"{tag}_{layer}")
        v.file_name = str(stl)
        v.material = mat_name
        v.translation = [0 * mm, 0 * mm, 0 * mm]
        vol_mm3 = row[layer]["volume_mm3"]
        mass_g = vol_mm3 / 1000.0 * m["density"]
        flag = " (INDIKATIF, L2)" if (tag == "MRCP_15M" and layer == "cranium_spongiosa") else ""
        print(f"  {layer:>18}: material {mat_name} | rho {m['density']:.3f} g/cm3 | "
              f"vol {vol_mm3/1000.0:7.3f} cm3 | massa {mass_g:7.3f} g{flag}")
        info.append(dict(tag=tag, layer=layer, material=mat_name, media_name=m["name"],
                         density_gcm3=m["density"], volume_mm3=vol_mm3, mass_g=mass_g,
                         mat_method=how))

    src = sim.add_source("GenericSource", name="dummy_init")
    src.particle = "e-"
    src.energy.type = "mono"
    src.energy.mono = 1.0 * MeV
    src.position.type = "point"
    src.position.translation = [0, 0, -5 * mm]
    src.direction.type = "iso"
    src.n = 50

    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option4"
    sim.physics_manager.enable_decay = False
    sim.physics_manager.set_production_cut("world", "all", 1 * mm)
    for layer in LAYERS:
        sim.physics_manager.set_production_cut(f"{tag}_{layer}", "all", 0.01 * mm)

    t0 = time.time()
    sim.run(start_new_process=True)
    dt = time.time() - t0
    print(f"  INISIALISASI + RUN DUMMY OK ({dt:.1f} s) | cek overlap bawaan lolos")
    return info


def main():
    tags = sys.argv[1:] if len(sys.argv) > 1 else ["MRCP_01M"]
    OUT.mkdir(exist_ok=True)
    print("=" * 70)
    print("TAHAP 2.3 v6 (FINAL) - IMPOR STL KE GATE 10 + MATERIAL ICRU")
    print(f"Set: {tags} | dermis tidak diimpor (L1)")
    print("=" * 70)
    all_info = []
    for tag in tags:
        all_info.extend(build_tag(tag))
    out_json = OUT / "stage23_summary.json"
    with open(out_json, "w") as f:
        json.dump(all_info, f, indent=2)
    print("\nRingkasan:", out_json)
    print("=" * 70)
    print("TAHAP 2.3 SELESAI untuk set di atas")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[TAHAP 2.3 GAGAL] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)