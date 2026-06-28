"""
Benchmark: Cartesian NOE + Cα chemical shift joint refinement of 2KJR (FR629A).

Protein
-------
N-terminal Ubiquitin-like (UBL) Domain from Tubulin-binding Cofactor B, CG11242,
from *Drosophila melanogaster*.
NESG target: FR629A (residues 8–92).
PDB: 2KJR  |  BMRB: 16338

Reference
---------
Ramelot, Cort, Shastry, Ciccosanti, Jiang, Nair, Rost, Swapna, Acton, Xiao,
Everett, Montelione, Kennedy.
*Solution NMR structure of the N-terminal Ubiquitin-like Domain from
Tubulin-binding Cofactor B.* To be published (deposited 2009).

NOESY peak lists used in Li, Spaman, Tejero, et al. (2023) *J. Magn. Reson.*
352:107481 (PMID 37257257) — the canonical reference for this data set.

Experimental data (from RCSB .mr file and BMRB 16338)
------------------------------------------------------
* 85 Cα chemical shifts (BMRB 16338, 96.8% backbone completeness).
* 1,536 NOE assign statements from 2KJR.mr (CNS/XPLOR format):
    - 832  sequential  (|i−j| < 2)
    - 255  medium-range (2 ≤ |i−j| < 5)
    - 449  long-range  (|i−j| ≥ 5)
  This benchmark uses the **Cα-traceable subset** (HA, N, HN, CA filtered
  to the backbone atom set), which provides ~200–350 restraints touching
  at least one backbone heavy atom.  Full sidechain atoms (HB*, HG*, etc.)
  are silently skipped because ``NOELoss`` currently operates on the N/Cα/C
  backbone coordinate array.

Strategy
--------
The NOE restraint set maps proton identities to the nearest resolved backbone
heavy atom using a conservative distance-addition:

    HA (Cα proton) → CA  (proton ~1.09 Å from Cα; upper bound relaxed by 1.1 Å)
    HN (amide H)   → N   (proton ~1.00 Å from N;  upper bound relaxed by 1.0 Å)

All other restraint atoms (HB*, HG*, HZ, HD*, etc.) are excluded.  This
limits coverage to restraints where at least *one* atom is HA or HN, and
the partner is also HA, HN, CA, or N.  Long-range HA↔HA restraints are the
most structurally informative class retained.

Design choices
--------------
1.  **Cartesian parameterization** — raw Cα backbone coordinates, no NeRF.
2.  **Annealed anchor** (``GeometryLoss``, weight 10→0.1 over 150 epochs):
    prevents non-physical backbone unravelling under the NOE potential.
3.  **BondLengthPenalty / BondAnglePenalty** (weights 50 / 10):
    Engh & Huber geometry enforcement.
4.  **ChiralityPenalty** (weight 20): guards Cα L→D inversions.
5.  **CartesianCAShiftLoss** (weight 1.0): 85 Cα shifts from BMRB 16338.
6.  **NOELoss** (weight 5.0, force_const 1.0): backbone-atom-traceable subset
    of 2KJR.mr.  Upper bounds extended by proton-to-heavy-atom offset where
    applicable.
7.  **EarlyStopping** on NOE term (patience 100) prevents overfit to
    sequential/trivial restraints once the structure has converged.
"""

import re
import sys
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from diff_biophys.geometry.backbone import (
    get_backbone_coords,
    get_residue_info,
    load_pdb_model,
)

from diff_integrator.loss import JointLoss
from diff_integrator.optimizer import EarlyStopping, IntegrativeRefiner
from diff_integrator.schedules import ExponentialDecaySchedule
from diff_integrator.terms.bond_geometry import make_backbone_bond_geometry
from diff_integrator.terms.chemical_shifts import CartesianCAShiftLoss
from diff_integrator.terms.chirality import make_backbone_chirality
from diff_integrator.terms.geometry import GeometryLoss
from diff_integrator.terms.noe import make_noe_restraints

# ---------------------------------------------------------------------------
# Data location
# ---------------------------------------------------------------------------

BENCH_DIR = Path("../diff-biophys/benchmarks/2KJR").resolve()
sys.path.insert(0, str(Path("../diff-biophys/benchmarks/HR2876B").resolve()))
from parse_nmrstar import load_bmrb_shifts  # noqa: E402

PDB_FILE      = BENCH_DIR / "2KJR.pdb"
BMRB_FILE     = BENCH_DIR / "bmrb16338_FR629A.str"
MR_FILE       = BENCH_DIR / "2KJR.mr"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

EPOCHS        = 2000
LEARNING_RATE = 0.005

WEIGHT_ANCHOR_INITIAL = 10.0
WEIGHT_ANCHOR_FINAL   = 0.1
WEIGHT_ANCHOR_DECAY   = 150       # τ epochs

WEIGHT_BOND      = 50.0
WEIGHT_ANGLE     = 10.0
WEIGHT_CHIRALITY = 20.0
WEIGHT_CA_SHIFTS = 1.0
WEIGHT_NOE       = 5.0
NOE_FORCE_CONST  = 1.0

LOG_INTERVAL = 50

# ---------------------------------------------------------------------------
# Atom-name → backbone heavy atom mapping
# ---------------------------------------------------------------------------

# Maps proton identity to the parent backbone heavy atom and a distance offset
# (Å) to add to the CNS upper bound.  Restraints not covered here are skipped.
_PROTON_TO_HEAVY = {
    "HA":  ("CA", 1.10),   # Cα–HA bond length ~1.09 Å
    "HN":  ("N",  1.02),   # N–HN bond length ~1.01 Å
    "H":   ("N",  1.02),   # alternate HN naming
    "CA":  ("CA", 0.00),   # already heavy
    "N":   ("N",  0.00),   # already heavy
    "C":   ("C",  0.00),   # already heavy (carbonyl C)
}


def _map_atom(atom: str) -> tuple[str, float] | None:
    """Return (heavy_atom_name, offset_Å) or None if atom is not backbone-traceable."""
    return _PROTON_TO_HEAVY.get(atom, None)


# ---------------------------------------------------------------------------
# CNS/XPLOR .mr file parser
# ---------------------------------------------------------------------------

_ASSIGN_RE = re.compile(
    r"assign\s*"
    r"\(\s*resid\s+(\d+)\s+and\s+name\s+(\S+)\s*\)"
    r"\s*"
    r"\(\s*resid\s+(\d+)\s+and\s+name\s+(\S+)\s*\)"
    r"\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
    re.IGNORECASE,
)


def parse_mr_noes(mr_path: Path) -> list[dict]:
    """Parse CNS/XPLOR NOE assign statements from a .mr file.

    Each ``assign`` line has the form::

        assign (resid R1 and name A1)(resid R2 and name A2)  d_minus  d  d_plus

    The CNS convention is:
        lower_bound = d − d_minus   (d_minus is often 0 for soft bounds)
        upper_bound = d + d_plus

    Returns a list of dicts with keys ``res_i``, ``atom_i``, ``res_j``,
    ``atom_j``, ``d_lower`` (Å), ``d_upper`` (Å) — **only** for pairs where
    both atoms are backbone-traceable (HA/HN/CA/N/C).
    """
    restraints = []
    skipped_atom = 0
    skipped_self = 0

    with open(mr_path) as f:
        text = f.read()

    # Remove continuation lines and collect all assign blocks
    text = text.replace("\n", " ")

    for m in _ASSIGN_RE.finditer(text):
        r1   = int(m.group(1))
        a1   = m.group(2)
        r2   = int(m.group(3))
        a2   = m.group(4)
        d_minus = float(m.group(5))
        d       = float(m.group(6))
        d_plus  = float(m.group(7))

        # Map to backbone heavy atoms
        heavy1 = _map_atom(a1)
        heavy2 = _map_atom(a2)

        if heavy1 is None or heavy2 is None:
            skipped_atom += 1
            continue

        heavy_name1, offset1 = heavy1
        heavy_name2, offset2 = heavy2

        # Same-atom restraints after mapping are trivial — skip
        if r1 == r2 and heavy_name1 == heavy_name2:
            skipped_self += 1
            continue

        lower = max(0.0, d - d_minus)
        upper = d + d_plus + max(offset1, offset2)

        restraints.append({
            "res_i":   r1,
            "atom_i":  heavy_name1,
            "res_j":   r2,
            "atom_j":  heavy_name2,
            "d_lower": lower,
            "d_upper": upper,
        })

    print(
        f"[parse_mr_noes] Total assign lines parsed: "
        f"{len(restraints) + skipped_atom + skipped_self}"
    )
    print(f"  → Backbone-traceable restraints retained: {len(restraints)}")
    print(f"  → Skipped (non-backbone atom):             {skipped_atom}")
    print(f"  → Skipped (self-pair after mapping):       {skipped_self}")
    return restraints


# ---------------------------------------------------------------------------
# BMRB Cα shift parser for NMR-STAR 3.1
# ---------------------------------------------------------------------------

def load_bmrb_ca_shifts_star(star_path: Path) -> dict[int, float]:
    """Parse Cα chemical shifts from an NMR-STAR 3.1 file.

    Returns a dict {residue_seq_id: Ca_shift_ppm}.
    """
    with open(star_path) as f:
        text = f.read()

    # Find the assigned_chem_shift_list_1 saveframe
    m = re.search(r"save_assigned_chem_shift_list_1(.*?)save_", text, re.DOTALL)
    if not m:
        raise RuntimeError(f"No assigned_chem_shift_list_1 saveframe in {star_path}")

    frame = m.group(1)
    lines = frame.split("\n")

    cols: list[str] = []
    in_loop = False
    shifts: dict[int, float] = {}

    seq_idx = atom_idx = val_idx = -1

    for line in lines:
        stripped = line.strip()
        if stripped == "loop_":
            in_loop = True
            cols = []
            seq_idx = atom_idx = val_idx = -1
            continue
        if in_loop and stripped.startswith("_Atom_chem_shift."):
            cols.append(stripped)
            continue
        if in_loop and stripped.startswith("stop_"):
            in_loop = False
            continue
        if in_loop and cols and stripped and not stripped.startswith("_"):
            # Assign column indices once we see data
            if seq_idx == -1:
                try:
                    seq_idx  = cols.index("_Atom_chem_shift.Seq_ID")
                    atom_idx = cols.index("_Atom_chem_shift.Atom_ID")
                    val_idx  = cols.index("_Atom_chem_shift.Val")
                except ValueError:
                    continue
            parts = stripped.split()
            if len(parts) > max(seq_idx, atom_idx, val_idx):
                if parts[atom_idx] == "CA":
                    try:
                        shifts[int(parts[seq_idx])] = float(parts[val_idx])
                    except ValueError:
                        pass

    return shifts


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_ca_rmsd(ca_shifts_obs: dict[int, float], coords: np.ndarray,
                    res_ids: np.ndarray) -> float:
    """Cα chemical shift RMSD (ppm) — proxy from structure using random-coil offset."""
    # This benchmark uses CartesianCAShiftLoss internally, which uses the
    # built-in shift predictor.  Here we compute the empirical RMSD from
    # the loss value (loss = mean squared deviation → RMSD = sqrt(loss)).
    raise NotImplementedError("Use CartesianCAShiftLoss.rms_deviation instead.")


def count_noe_violations(noe_loss, coords_jnp: jnp.ndarray) -> dict:
    """Delegate to NOELoss.count_violations."""
    return noe_loss.count_violations(coords_jnp)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Benchmark: 2KJR (FR629A) — NOE + Cα shifts, Cartesian refinement")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load PDB structure (Model 1)
    # ------------------------------------------------------------------
    print(f"\n[1] Loading PDB: {PDB_FILE}")
    pdb_model = load_pdb_model(str(PDB_FILE), model_id=1)
    res_ids_arr, _res_names = get_residue_info(pdb_model)  # tuple (res_ids, res_names)
    coords0 = get_backbone_coords(pdb_model)               # (N_atoms, 3) flat N/CA/C
    res_ids = np.asarray(res_ids_arr)
    n_res   = len(res_ids)
    print(f"    {n_res} residues (seq IDs {res_ids[0]}–{res_ids[-1]})")
    print(f"    coords shape: {coords0.shape}")

    coords_flat0 = coords0  # already (N_atoms, 3) — N/CA/C per residue

    # ------------------------------------------------------------------
    # 2. Load Cα chemical shifts
    # ------------------------------------------------------------------
    print(f"\n[2] Loading Cα shifts from BMRB 16338: {BMRB_FILE}")
    ca_shifts = load_bmrb_ca_shifts_star(BMRB_FILE)
    print(f"    {len(ca_shifts)} Cα shifts (residues {min(ca_shifts)} – {max(ca_shifts)})")

    # Build observed-shifts array aligned to res_ids
    obs_shifts = np.full(n_res, np.nan)
    for i, rid in enumerate(res_ids):
        if rid in ca_shifts:
            obs_shifts[i] = ca_shifts[rid]
    n_matched = int(np.sum(~np.isnan(obs_shifts)))
    print(f"    {n_matched} residues matched to structure")

    # ------------------------------------------------------------------
    # 3. Parse NOE restraints from .mr file
    # ------------------------------------------------------------------
    print(f"\n[3] Parsing NOE restraints from: {MR_FILE}")
    raw_noes = parse_mr_noes(MR_FILE)

    # Filter to residues present in structure
    res_set = set(int(r) for r in res_ids)
    filtered_noes = [
        noe for noe in raw_noes
        if noe["res_i"] in res_set and noe["res_j"] in res_set
    ]
    # Range breakdown
    lr = sum(1 for n in filtered_noes if abs(n["res_i"] - n["res_j"]) >= 5)
    mr_ = sum(1 for n in filtered_noes if 2 <= abs(n["res_i"] - n["res_j"]) < 5)
    sr = sum(1 for n in filtered_noes if abs(n["res_i"] - n["res_j"]) < 2)
    print(f"    After residue filter: {len(filtered_noes)} restraints retained")
    print(f"      Sequential (|i−j|<2):   {sr}")
    print(f"      Medium (2≤|i−j|<5):     {mr_}")
    print(f"      Long-range (|i−j|≥5):   {lr}")

    if len(filtered_noes) == 0:
        raise RuntimeError(
            "No backbone-traceable NOE restraints survived filtering.  "
            "Check residue numbering in 2KJR.mr vs PDB structure."
        )

    # ------------------------------------------------------------------
    # 4. Build NOELoss
    # ------------------------------------------------------------------
    print(f"\n[4] Building NOELoss ({len(filtered_noes)} restraints)...")
    try:
        noe_loss = make_noe_restraints(
            noe_list=filtered_noes,
            res_ids=res_ids,
            atom_names=["N", "CA", "C"],
            force_const=NOE_FORCE_CONST,
        )
    except ValueError as exc:
        print(f"  [WARNING] Some restraints could not be mapped: {exc}")
        # Re-filter strictly to CA-to-CA and N-to-N restraints only
        ca_ca_noes = [
            n for n in filtered_noes
            if n["atom_i"] in ("CA",) and n["atom_j"] in ("CA",)
        ]
        print(f"  Falling back to {len(ca_ca_noes)} CA–CA restraints only.")
        noe_loss = make_noe_restraints(
            noe_list=ca_ca_noes,
            res_ids=res_ids,
            atom_names=["N", "CA", "C"],
            force_const=NOE_FORCE_CONST,
        )

    # Baseline violation count
    coords_jnp0    = jnp.array(coords_flat0)
    baseline_viols = noe_loss.count_violations(coords_jnp0)
    baseline_rms   = noe_loss.rms_violation(coords_jnp0)
    print(f"    Baseline violations: {baseline_viols['total']} "
          f"(upper={baseline_viols['upper']}, lower={baseline_viols['lower']})")
    print(f"    Baseline RMS violation: {baseline_rms:.3f} Å")

    # ------------------------------------------------------------------
    # 5. Build CartesianCAShiftLoss
    # ------------------------------------------------------------------
    print("\n[5] Building CartesianCAShiftLoss...")
    # Build aligned shift arrays
    valid_mask = ~np.isnan(obs_shifts)
    shift_res_ids = res_ids[valid_mask].astype(np.int32)
    shift_obs     = obs_shifts[valid_mask].astype(np.float32)
    ca_shift_loss = CartesianCAShiftLoss(
        exp_res_ids=shift_res_ids,
        exp_shifts=shift_obs,
        struct_res_ids=res_ids.astype(np.int32),
        struct_res_names=list(_res_names),
    )
    # Baseline RMSD from shift loss
    baseline_shift_loss = float(ca_shift_loss(None, coords_jnp0))
    print(f"    Baseline Cα shift loss (MSE): {baseline_shift_loss:.4f} ppm²")
    print(f"    Baseline Cα shift RMSD: {baseline_shift_loss**0.5:.4f} ppm")

    # ------------------------------------------------------------------
    # 6. Build geometry terms
    # ------------------------------------------------------------------
    print("\n[6] Building geometry terms...")
    anchor_loss   = GeometryLoss(target_coords=coords_jnp0)
    bond_penalty, angle_penalty = make_backbone_bond_geometry(n_res)
    chirality_penalty = make_backbone_chirality(n_res)

    # ------------------------------------------------------------------
    # 7. Assemble JointLoss
    # ------------------------------------------------------------------
    #   Term 0: anchor        (GeometryLoss, annealed)
    #   Term 1: bond penalty  (BondLengthPenalty)
    #   Term 2: angle penalty (BondAnglePenalty)
    #   Term 3: chirality     (ChiralityPenalty)
    #   Term 4: Ca shifts     (CartesianCAShiftLoss)  ← EarlyStopping target
    #   Term 5: NOE           (NOELoss)
    joint_loss = JointLoss(
        terms=[
            (anchor_loss,       WEIGHT_ANCHOR_INITIAL),
            (bond_penalty,      WEIGHT_BOND),
            (angle_penalty,     WEIGHT_ANGLE),
            (chirality_penalty, WEIGHT_CHIRALITY),
            (ca_shift_loss,     WEIGHT_CA_SHIFTS),
            (noe_loss,          WEIGHT_NOE),
        ]
    )

    # ------------------------------------------------------------------
    # 8. Anchor schedule and refiner
    # ------------------------------------------------------------------
    anchor_schedule = ExponentialDecaySchedule(
        initial_weight=WEIGHT_ANCHOR_INITIAL,
        final_weight=WEIGHT_ANCHOR_FINAL,
        decay_epochs=WEIGHT_ANCHOR_DECAY,
    )

    refiner = IntegrativeRefiner(
        loss_fn=joint_loss,
    )

    # ------------------------------------------------------------------
    # 9. Per-epoch monitoring callback
    # ------------------------------------------------------------------
    def monitor(epoch: int, coords: jnp.ndarray, _loss: float) -> None:
        if epoch % LOG_INTERVAL != 0:
            return
        viols = noe_loss.count_violations(coords)
        rms_v = noe_loss.rms_violation(coords)
        sl    = float(ca_shift_loss(None, coords))
        print(
            f"  [epoch {epoch:5d}]  "
            f"NOE violations: {viols['total']:4d}  "
            f"RMS viol: {rms_v:.3f} Å  |  "
            f"Cα shift RMSD: {sl**0.5:.4f} ppm"
        )

    # ------------------------------------------------------------------
    # 10. Run refinement
    # ------------------------------------------------------------------
    print(f"\n[7] Running refinement ({EPOCHS} epochs max, lr={LEARNING_RATE})...")
    print("    EarlyStopping on Ca shift term (index 4, patience=100)")
    print()

    result = refiner.run(
        init_params=coords_jnp0,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        kinematics_fn=None,          # Cartesian mode
        weight_schedules={0: anchor_schedule},
        per_epoch_callbacks=[monitor],
        early_stopping=EarlyStopping(term_index=4, patience=100),
    )

    coords_final = result.final_params

    # ------------------------------------------------------------------
    # 11. Final evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # NOE violations
    viols_final = noe_loss.count_violations(coords_final)
    rms_v_final = noe_loss.rms_violation(coords_final)
    print(f"\nNOE restraints ({noe_loss.n_restraints} backbone-traceable):")
    print(f"  Violations BEFORE: {baseline_viols['total']:4d}  "
          f"(RMS {baseline_rms:.3f} Å)")
    print(f"  Violations AFTER:  {viols_final['total']:4d}  "
          f"(RMS {rms_v_final:.3f} Å)")
    print(f"  Improvement: {baseline_viols['total'] - viols_final['total']} fewer violations")

    # Cα shifts
    sl_final = float(ca_shift_loss(None, coords_final))
    print(f"\nCα chemical shifts ({n_matched} residues):")
    print(f"  RMSD BEFORE: {baseline_shift_loss**0.5:.4f} ppm")
    print(f"  RMSD AFTER:  {sl_final**0.5:.4f} ppm")
    delta_rmsd = baseline_shift_loss**0.5 - sl_final**0.5
    print(f"  Improvement: Δ = {delta_rmsd:+.4f} ppm")

    # Structural drift from NMR Model 1
    coords_final_np = np.array(coords_final)
    ca_before = coords_flat0[1::3]    # every CA (index 1 in N/CA/C triplet)
    ca_after  = coords_final_np[1::3]
    structural_rmsd = float(np.sqrt(np.mean(np.sum((ca_after - ca_before)**2, axis=-1))))
    print(f"\nStructural Cα RMSD vs NMR Model 1: {structural_rmsd:.3f} Å")
    print(f"Stopped at epoch: {result.stopped_at_epoch}")

    # Save outputs
    out_dir = Path("results/2KJR_cartesian")
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "noe_violations_before.npy", np.array([baseline_viols['total']]))
    np.save(out_dir / "noe_violations_after.npy",  np.array([viols_final['total']]))
    np.save(out_dir / "noe_rms_before.npy",        np.array([baseline_rms]))
    np.save(out_dir / "noe_rms_after.npy",         np.array([rms_v_final]))
    np.save(out_dir / "ca_shift_rmsd_before.npy",  np.array([baseline_shift_loss**0.5]))
    np.save(out_dir / "ca_shift_rmsd_after.npy",   np.array([sl_final**0.5]))
    np.save(out_dir / "structural_rmsd.npy",        np.array([structural_rmsd]))
    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
