"""
parse_nmrstar.py — Universal NMR-STAR v3 parser for diff-biophys benchmarks.

Handles both chemical shift saveframes and RDC saveframes, which use
different tag namespaces (_Atom_chem_shift.* vs _RDC.*).

Exports:
    load_bmrb_shifts(path)   -> dict keyed by atom type (CA, N, H)
    load_bmrb_rdcs(path)     -> dict keyed by saveframe name, each with
                                 res_id, atom1, atom2, rdc arrays
"""

from pathlib import Path
from typing import NamedTuple

import numpy as np

# ── Shared helpers ────────────────────────────────────────────────────────────


def _extract_saveframe(lines: list[str], frame_name: str) -> list[str]:
    """Return lines between 'save_<frame_name>' and the closing 'save_'."""
    start = None
    for i, line in enumerate(lines):
        if line.strip() == f"save_{frame_name}":
            start = i + 1
            break
    if start is None:
        raise ValueError(f"Saveframe 'save_{frame_name}' not found.")
    frame = []
    for line in lines[start:]:
        if line.strip() == "save_":
            break
        frame.append(line)
    return frame


def _find_loop(frame_lines: list[str], tag_prefix: str) -> tuple[list[str], int]:
    """
    Find the loop_ containing columns starting with tag_prefix.
    Returns (column_names_without_prefix, data_start_index).
    """
    for i, line in enumerate(frame_lines):
        if line.strip() != "loop_":
            continue
        # Check if next non-empty line has the expected prefix
        headers = []
        data_start = None
        for j in range(i + 1, len(frame_lines)):
            stripped = frame_lines[j].strip()
            if stripped.startswith(tag_prefix):
                headers.append(stripped.split(".")[-1])
            elif headers and stripped and not stripped.startswith("_"):
                data_start = j
                break
            elif not stripped:
                continue
            elif headers and stripped.startswith("_") and not stripped.startswith(tag_prefix):
                # Different tag prefix — not our loop
                break
        if headers and data_start is not None:
            return headers, data_start
    raise ValueError(f"Loop with tag prefix '{tag_prefix}' not found in saveframe.")


def _parse_loop_data(frame_lines: list[str], data_start: int, n_cols: int) -> list[list[str]]:
    """Parse data rows from a loop, stopping at 'stop_' or a new saveframe."""
    rows = []
    for line in frame_lines[data_start:]:
        stripped = line.strip()
        if stripped in ("stop_", "loop_") or stripped.startswith("save_"):
            break
        if not stripped:
            continue
        tokens = stripped.split()
        if len(tokens) >= n_cols:
            rows.append(tokens)
    return rows


# ── Chemical shifts ───────────────────────────────────────────────────────────


class ShiftEntry(NamedTuple):
    res_id: int
    res_name: str
    atom: str
    shift: float
    error: float


def parse_chem_shifts(path: Path) -> list[ShiftEntry]:
    """
    Parse _Atom_chem_shift loop from NMR-STAR v3.
    Finds the 'save_assigned_chem_shift_list_1' saveframe.
    """
    lines = path.read_text().splitlines()
    frame = _extract_saveframe(lines, "assigned_chem_shift_list_1")
    headers, data_start = _find_loop(frame, "_Atom_chem_shift.")
    col = {h: i for i, h in enumerate(headers)}
    required = {"Comp_index_ID", "Comp_ID", "Atom_ID", "Val", "Val_err"}
    if not required.issubset(col):
        raise ValueError(f"Missing columns: {required - set(col)}")

    entries = []
    for row in _parse_loop_data(frame, data_start, len(headers)):
        try:
            val_str = row[col["Val"]]
            if val_str in (".", "?"):
                continue
            entries.append(
                ShiftEntry(
                    res_id=int(row[col["Comp_index_ID"]]),
                    res_name=row[col["Comp_ID"]],
                    atom=row[col["Atom_ID"]],
                    shift=float(val_str),
                    error=float(row[col["Val_err"]])
                    if row[col["Val_err"]] not in (".", "?")
                    else 0.0,
                )
            )
        except (ValueError, IndexError):
            continue
    return entries


def load_bmrb_shifts(path: Path | str) -> dict[str, dict]:
    """
    Load backbone chemical shifts (CA, N, HN) from NMR-STAR v3.
    Returns dict keyed by atom type:
        {"CA": {"res_id": np.array, "res_name": list, "shift": np.array, "error": np.array}}
    """
    path = Path(path)
    entries = parse_chem_shifts(path)
    result: dict[str, dict] = {}
    for atom_type in ("CA", "N", "H"):
        subset = [e for e in entries if e.atom == atom_type]
        if subset:
            result[atom_type] = {
                "res_id": np.array([e.res_id for e in subset], dtype=np.int32),
                "res_name": [e.res_name for e in subset],
                "shift": np.array([e.shift for e in subset], dtype=np.float32),
                "error": np.array([e.error for e in subset], dtype=np.float32),
            }
    return result


# ── RDC data ─────────────────────────────────────────────────────────────────


class RDCEntry(NamedTuple):
    res_id: int  # residue number (Comp_index_ID of atom 1, typically N)
    res_name: str  # 3-letter code of residue
    atom1: str  # e.g. "N"
    atom2: str  # e.g. "H"
    rdc_hz: float  # measured value in Hz
    error_hz: float  # uncertainty in Hz (0.0 if not provided)


def _find_rdc_saveframes(lines: list[str]) -> list[str]:
    """Return names of all RDC saveframes (those where Sf_category == 'RDCs')."""
    names = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("save_") and stripped != "save_":
            # Peek ahead to see if it's an RDC saveframe
            for peek in lines[i + 1 : i + 20]:
                if "_RDC_list.Sf_category" in peek and "RDCs" in peek:
                    names.append(stripped[len("save_") :])
                    break
                if "save_" in peek and peek.strip() != "save_":
                    break
    return names


def parse_rdc_saveframe(lines: list[str], frame_name: str) -> list[RDCEntry]:
    """Parse a single RDC saveframe by name."""
    frame = _extract_saveframe(lines, frame_name)
    headers, data_start = _find_loop(frame, "_RDC.")
    col = {h: i for i, h in enumerate(headers)}

    required = {"Comp_index_ID_1", "Comp_ID_1", "Atom_ID_1", "Atom_ID_2", "Val"}
    if not required.issubset(col):
        raise ValueError(f"Missing required RDC columns in saveframe '{frame_name}'.")

    entries = []
    for row in _parse_loop_data(frame, data_start, len(headers)):
        try:
            val_str = row[col["Val"]]
            if val_str in (".", "?"):
                continue
            atom1 = row[col["Atom_ID_1"]]
            atom2 = row[col["Atom_ID_2"]]
            # Only extract ¹⁵N-¹H (N-H) RDCs
            if not ({"N", "H"} == {atom1, atom2}):
                continue
            # Make sure atom1 is N (the heavier atom)
            if atom1 == "H":
                res_id_key = "Comp_index_ID_2"
                res_name_key = "Comp_ID_2"
            else:
                res_id_key = "Comp_index_ID_1"
                res_name_key = "Comp_ID_1"
            err_str = row[col["Val_err"]] if "Val_err" in col else "."
            entries.append(
                RDCEntry(
                    res_id=int(row[col[res_id_key]]),
                    res_name=row[col[res_name_key]],
                    atom1="N",
                    atom2="H",
                    rdc_hz=float(val_str),
                    error_hz=float(err_str) if err_str not in (".", "?") else 0.5,
                )
            )
        except (ValueError, IndexError):
            continue
    return entries


def load_bmrb_rdcs(path: Path | str) -> dict[str, dict]:
    """
    Load all ¹⁵N-¹H RDC saveframes from NMR-STAR v3.
    Returns dict keyed by saveframe name:
        {"RDC_list_1": {"res_id": np.array, "res_name": list, "rdc": np.array, "error": np.array}}
    Saveframe names are the raw NMR-STAR framecodes (e.g. "RDC_list_1", "RDC_phage").
    """
    path = Path(path)
    lines = path.read_text().splitlines()
    frame_names = _find_rdc_saveframes(lines)
    if not frame_names:
        raise ValueError(f"No RDC saveframes found in {path.name}.")

    result: dict[str, dict] = {}
    for name in frame_names:
        try:
            entries = parse_rdc_saveframe(lines, name)
            if entries:
                result[name] = {
                    "res_id": np.array([e.res_id for e in entries], dtype=np.int32),
                    "res_name": [e.res_name for e in entries],
                    "rdc": np.array([e.rdc_hz for e in entries], dtype=np.float32),
                    "error": np.array([e.error_hz for e in entries], dtype=np.float32),
                }
        except (ValueError, KeyError) as exc:
            print(f"  Warning: could not parse RDC saveframe '{name}': {exc}")
    return result


if __name__ == "__main__":
    import sys

    path = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "bmrb16746_GmR58A.str"
    )
    print(f"Parsing: {path.name}")

    shifts = load_bmrb_shifts(path)
    for atom, d in shifts.items():
        if d:
            print(
                f"  {atom}: {len(d['res_id'])} residues, {d['shift'].min():.1f}–{d['shift'].max():.1f} ppm"
            )

    rdcs = load_bmrb_rdcs(path)
    for name, d in rdcs.items():
        print(
            f"  RDC '{name}': {len(d['res_id'])} ¹⁵N-¹H couplings, "
            f"{d['rdc'].min():.1f}–{d['rdc'].max():.1f} Hz"
        )
