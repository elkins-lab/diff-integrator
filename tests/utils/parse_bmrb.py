"""
parse_bmrb.py — Extract chemical shift data from BMRB NMR-STAR files.

Provides structured dicts for each backbone atom type (CA, N, H).
This module handles the BMRB-specific NMR-STAR v3 format; it is intentionally
kept separate from the main diff_biophys package because NMR-STAR is a
specialised deposition format, not a general interchange format.

Note: generic RDC table loading (whitespace-delimited TSV/DC format) has moved
to :func:`diff_biophys.nmr.io.load_rdc_table`.

Usage:
    data = load_bmrb_shifts("bmrb17020.str")
    ca = data["CA"]   # {"res_id": np.array, "res_name": list, "shift": np.array}
"""

from pathlib import Path
from typing import NamedTuple

import numpy as np


class ShiftEntry(NamedTuple):
    res_id: int
    res_name: str
    atom: str
    shift: float
    error: float


def parse_nmrstar_chem_shifts(path: Path) -> list[ShiftEntry]:
    """
    Parse the _Atom_chem_shift loop from an NMR-STAR v3 file.
    Handles any number of columns by reading the header dynamically.
    """
    text = path.read_text()
    lines = text.splitlines()

    # --- Locate the assigned_chem_shift_list saveframe ---
    saveframe_start = None
    for i, line in enumerate(lines):
        if line.strip() == "save_assigned_chem_shift_list_1":
            saveframe_start = i
            break
    if saveframe_start is None:
        raise ValueError("Cannot find 'save_assigned_chem_shift_list_1' in NMR-STAR file.")

    # Collect lines until the closing "save_" sentinel
    frame_lines = []
    for line in lines[saveframe_start + 1 :]:
        if line.strip() == "save_":
            break
        frame_lines.append(line)

    # --- Find the _Atom_chem_shift loop ---
    loop_start = None
    for i, line in enumerate(frame_lines):
        if line.strip() == "loop_":
            # Peek at next non-empty line
            for j in range(i + 1, min(i + 5, len(frame_lines))):
                peek = frame_lines[j].strip()
                if peek.startswith("_Atom_chem_shift."):
                    loop_start = i
                    break
        if loop_start is not None:
            break

    if loop_start is None:
        raise ValueError("Cannot find '_Atom_chem_shift loop_' in saveframe.")

    # --- Parse column headers ---
    headers = []
    data_start = None
    for i, line in enumerate(frame_lines[loop_start + 1 :], start=loop_start + 1):
        stripped = line.strip()
        if stripped.startswith("_Atom_chem_shift."):
            headers.append(stripped.split(".")[-1])
        elif stripped == "stop_":
            break
        elif headers and stripped and not stripped.startswith("_"):
            data_start = i
            break

    if not headers or data_start is None:
        raise ValueError("Could not parse _Atom_chem_shift headers/data.")

    # Build column index map
    col = {h: i for i, h in enumerate(headers)}

    required = {"Comp_index_ID", "Comp_ID", "Atom_ID", "Val", "Val_err"}
    missing = required - set(col.keys())
    if missing:
        raise ValueError(f"Missing required columns in NMR-STAR file: {missing}")

    # --- Parse data rows ---
    entries = []
    for line in frame_lines[data_start:]:
        stripped = line.strip()
        if stripped == "stop_" or stripped.startswith("_") or stripped == "loop_":
            break
        if not stripped:
            continue
        tokens = stripped.split()
        if len(tokens) < len(headers):
            continue
        try:
            res_id = int(tokens[col["Comp_index_ID"]])
            res_name = tokens[col["Comp_ID"]]
            atom_id = tokens[col["Atom_ID"]]
            val_str = tokens[col["Val"]]
            err_str = tokens[col["Val_err"]]

            if val_str in (".", "?"):
                continue
            shift = float(val_str)
            error = float(err_str) if err_str not in (".", "?") else 0.0
            entries.append(ShiftEntry(res_id, res_name, atom_id, shift, error))
        except (ValueError, IndexError):
            continue

    return entries


def load_bmrb_shifts(path: Path | str) -> dict:
    """
    Load BMRB NMR-STAR chemical shifts grouped by backbone atom type.

    Returns:
        dict with keys "CA", "N", "H" (amide HN), each:
            {
                "res_id":   np.array[int],
                "res_name": list[str],
                "shift":    np.array[float],
                "error":    np.array[float],
            }
    """
    path = Path(path)
    all_entries = parse_nmrstar_chem_shifts(path)

    result: dict[str, dict] = {}
    for atom_type in ("CA", "N", "H"):
        subset = [e for e in all_entries if e.atom == atom_type]
        if subset:
            result[atom_type] = {
                "res_id": np.array([e.res_id for e in subset], dtype=np.int32),
                "res_name": [e.res_name for e in subset],
                "shift": np.array([e.shift for e in subset], dtype=np.float32),
                "error": np.array([e.error for e in subset], dtype=np.float32),
            }
    return result


if __name__ == "__main__":
    bmrb_path = Path(__file__).parent / "bmrb17020.str"
    print(f"Parsing: {bmrb_path}")
    data = load_bmrb_shifts(bmrb_path)

    for atom_type, d in data.items():
        print(
            f"\n{atom_type}: {len(d['res_id'])} residues "
            f"({d['shift'].min():.2f}–{d['shift'].max():.2f} ppm)"
        )
        for i in range(min(8, len(d["res_id"]))):
            print(f"  [{d['res_id'][i]:3d}] {d['res_name'][i]:3s}  {d['shift'][i]:.3f} ppm")
