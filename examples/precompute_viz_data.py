"""
Pre-compute static data arrays needed by the visualization notebooks.

Run from the diff-integrator repo root:
    PYTHONPATH=. python examples/precompute_viz_data.py

Outputs per protein (benchmarks/results/<name>/):
    ca_shifts_before.npy   predicted Ca shifts from initial structure
    ca_shifts_after.npy    predicted Ca shifts from final structure
    ca_shifts_exp.npy      experimental Ca shifts (aligned to res_ids order)
    ca_shifts_res_ids.npy  residue IDs array
    nerf_drift.npy         Kabsch RMSD of NeRF-rebuilt vs raw PDB Ca atoms

2KZV only:
    phi_psi_init.npy       (N,2) initial phi/psi in degrees for Ramachandran
    rdc_pag_exp.npy        experimental PAG RDCs (Hz)
    rdc_peg_exp.npy        experimental PEG RDCs (Hz)
    rdc_pag_q_before.npy   Q-factor from initial coords
    rdc_pag_q_after.npy    Q-factor from final coords
    rdc_peg_q_before.npy   Q-factor from initial coords
    rdc_peg_q_after.npy    Q-factor from final coords
"""

import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp

REPO = Path(".").resolve()
BIOPHYS = (Path("..") / "diff-biophys").resolve()

sys.path.insert(0, str(REPO))

from diff_biophys.geometry.backbone import (
    compute_phi_psi,
    get_backbone_coords,
    get_residue_info,
    load_pdb_model,
    make_backbone_builder,
)
from diff_biophys.nmr.chemical_shifts import predict_ca_shifts
import biotite.structure.io.pdb as pdb_io
from diff_biophys.nmr.chemical_shifts import RANDOM_COIL_CA


# ─── helpers ────────────────────────────────────────────────────────────────

def kabsch_rmsd(A: np.ndarray, B: np.ndarray) -> float:
    A = np.array(A).reshape(-1, 3)
    B = np.array(B).reshape(-1, 3)
    A = A - A.mean(0)
    B = B - B.mean(0)
    U, _, Vt = np.linalg.svd(A.T @ B)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = U @ Vt
    return float(np.sqrt(np.mean(np.sum((A @ R - B) ** 2, axis=1))))


def load_pdb_ca(path: Path) -> np.ndarray:
    """Return Cα coordinates from a PDB file (all atoms)."""
    f = pdb_io.PDBFile.read(str(path))
    atoms = pdb_io.get_structure(f, model=1)
    return atoms[atoms.atom_name == "CA"].coord


def load_pdb_all(path: Path) -> np.ndarray:
    """Return all backbone atom coordinates (N, CA, C) from a PDB file."""
    f = pdb_io.PDBFile.read(str(path))
    atoms = pdb_io.get_structure(f, model=1)
    return atoms.coord.reshape(-1, 3)


# ─── protein definitions ────────────────────────────────────────────────────

PROTEINS = [
    {
        "name": "2KZV",
        "bench_dir": BIOPHYS / "benchmarks/2KZV",
        "pdb": "2KZV.pdb",
        "bmrb_module": str(BIOPHYS / "benchmarks/2KZV"),
        "bmrb_file": "bmrb17020.str",
        "bmrb_fn": "parse_bmrb:load_bmrb_shifts",
        "has_rdc": True,
    },
    {
        "name": "GmR58A",
        "bench_dir": BIOPHYS / "benchmarks/GmR58A",
        "pdb": "2KUT.pdb",
        "bmrb_module": str(BIOPHYS / "benchmarks/GmR58A"),
        "bmrb_file": "bmrb16746_GmR58A.str",
        "bmrb_fn": "parse_nmrstar:load_bmrb_shifts",
        "has_rdc": False,
    },
    {
        "name": "HR2876B",
        "bench_dir": BIOPHYS / "benchmarks/HR2876B",
        "pdb": "2LTM.pdb",
        "bmrb_module": str(BIOPHYS / "benchmarks/HR2876B"),
        "bmrb_file": "bmrb18489_HR2876B.str",
        "bmrb_fn": "parse_nmrstar:load_bmrb_shifts",
        "has_rdc": False,
    },
]


# ─── main loop ──────────────────────────────────────────────────────────────

for p in PROTEINS:
    name = p["name"]
    print(f"\n{'='*50}")
    print(f" Processing {name}")
    print(f"{'='*50}")

    bd = p["bench_dir"]
    results_dir = REPO / f"benchmarks/results/{name}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load parse module
    sys.path.insert(0, p["bmrb_module"])
    mod_name, fn_name = p["bmrb_fn"].split(":")
    mod = __import__(mod_name)
    load_bmrb_shifts = getattr(mod, fn_name)

    # Load structure
    struct = load_pdb_model(bd / p["pdb"], model_id=1)
    res_ids, res_names = get_residue_info(struct)
    coords = get_backbone_coords(struct)
    n_res = len(res_ids)
    build_bb = make_backbone_builder(n_res, coords[:3])
    init_phi, init_psi = compute_phi_psi(coords)
    print(f"  {n_res} residues")

    # NeRF-rebuilt initial coordinates
    nerf_init = build_bb(init_phi, init_psi)
    nerf_ca = np.array(nerf_init).reshape(-1, 3)[1::3]   # CA is index 1 in N,CA,C

    # Raw PDB Cα
    raw_ca = load_pdb_ca(bd / p["pdb"])
    n_match = min(len(nerf_ca), len(raw_ca))
    drift = kabsch_rmsd(nerf_ca[:n_match], raw_ca[:n_match])
    np.save(results_dir / "nerf_drift.npy", np.array(drift))
    print(f"  NeRF drift: {drift:.3f} Å")

    # Chemical shift prediction
    bmrb = load_bmrb_shifts(bd / p["bmrb_file"])
    ca_data = bmrb["CA"]
    exp_res_ids = np.array(ca_data["res_id"])
    exp_shifts = np.array(ca_data["shift"])

    # Build residue-code → random coil shift lookup
    res_name_map = dict(zip(res_ids, res_names))
    rc_shifts = np.array([
        RANDOM_COIL_CA.get(res_name_map.get(r, "GLY"), 56.0)
        for r in res_ids
    ])

    # Predict from initial (NeRF)
    pred_before = np.array(predict_ca_shifts(init_phi, init_psi, jnp.array(rc_shifts)))
    np.save(results_dir / "ca_shifts_before.npy", pred_before)

    # Predict from final (load final PDB, extract phi/psi)
    final_all = load_pdb_all(results_dir / "final.pdb")
    # Recompute phi/psi from the saved final backbone
    final_phi, final_psi = compute_phi_psi(jnp.array(final_all))
    pred_after = np.array(predict_ca_shifts(final_phi, final_psi, jnp.array(rc_shifts)))
    np.save(results_dir / "ca_shifts_after.npy", pred_after)

    # Experimental shifts aligned to res_ids order
    exp_dict = dict(zip(exp_res_ids.tolist(), exp_shifts.tolist()))
    aligned_exp = np.array([exp_dict.get(int(r), np.nan) for r in res_ids])
    np.save(results_dir / "ca_shifts_exp.npy", aligned_exp)
    np.save(results_dir / "ca_shifts_res_ids.npy", np.array(res_ids))

    n_matched = int(np.sum(~np.isnan(aligned_exp)))
    rmsd_before = float(np.sqrt(np.nanmean((pred_before - aligned_exp) ** 2)))
    rmsd_after = float(np.sqrt(np.nanmean((pred_after - aligned_exp) ** 2)))
    print(f"  Cα shifts matched: {n_matched}")
    print(f"  RMSD before: {rmsd_before:.3f} ppm  →  after: {rmsd_after:.3f} ppm")

    # Phi/psi for Ramachandran (2KZV only)
    if name == "2KZV":
        phi_deg = np.degrees(np.array(init_phi))
        psi_deg = np.degrees(np.array(init_psi))
        np.save(results_dir / "phi_psi_init.npy",
                np.stack([phi_deg, psi_deg], axis=1))
        print(f"  Saved phi/psi ({len(phi_deg)} residues)")


# ─── 2KZV RDC data ──────────────────────────────────────────────────────────

print(f"\n{'='*50}")
print(" 2KZV — RDC Q-factors")
print(f"{'='*50}")

from diff_biophys.nmr.io import load_rdc_table
from diff_biophys.nmr.rdc import make_rdc_refinement_fns

BENCH_2KZV = BIOPHYS / "benchmarks/2KZV"
results_2kzv = REPO / "benchmarks/results/2KZV"

struct = load_pdb_model(BENCH_2KZV / "2KZV.pdb", model_id=1)
res_ids, res_names = get_residue_info(struct)
coords = get_backbone_coords(struct)
build_bb = make_backbone_builder(len(res_ids), coords[:3])
init_phi, init_psi = compute_phi_psi(coords)
init_coords = build_bb(init_phi, init_psi)

final_all = load_pdb_all(results_2kzv / "final.pdb")
final_phi, final_psi = compute_phi_psi(jnp.array(final_all))
final_coords = build_bb(final_phi, final_psi)

for medium in ("PAG", "PEG"):
    rdc_data = load_rdc_table(BENCH_2KZV / f"rdc_{medium}.tsv")[medium]
    exp_rdcs = np.array(rdc_data["rdc"])
    _, q_fn, _, _ = make_rdc_refinement_fns(
        rdc_data["res_id"], rdc_data["rdc"], res_ids
    )
    q_before = float(q_fn(init_coords))
    q_after = float(q_fn(final_coords))
    np.save(results_2kzv / f"rdc_{medium.lower()}_exp.npy", exp_rdcs)
    np.save(results_2kzv / f"rdc_{medium.lower()}_q_before.npy", np.array(q_before))
    np.save(results_2kzv / f"rdc_{medium.lower()}_q_after.npy", np.array(q_after))
    print(f"  {medium}: Q before={q_before:.3f} → after={q_after:.3f}")

print("\n✓ All pre-computation complete!")
