from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Union, Optional

import pandas as pd


def _decode_bytes(data: Union[bytes, str]) -> str:
    """Decode bytes to text using utf-8 with fallbacks.

    Tries utf-8 first, then latin-1, and finally ignores errors.
    If input is already str, just converts to str (no-op).
    """
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return data.decode("latin-1")
        except UnicodeDecodeError:
            return data.decode("utf-8", errors="ignore")
    except AttributeError:
        return str(data)


def _iter_lines(src: Union[str, os.PathLike, bytes, Iterable[str]]) -> Iterable[str]:
    """Yield lines from different input types as text strings.

    Supported inputs:
    - str/Path: treated as a filesystem path and read
    - bytes: decoded to text
    - Iterable[str]: yielded as-is
    - str containing newlines and '>': treated as FASTA text (not a path)
    """
    if isinstance(src, (str, Path)):
        p = Path(src)
        if p.exists():
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    yield line.rstrip("\n\r")
            return
        # If it looks like FASTA text content rather than a path
        text = str(src)
        for line in text.splitlines():
            yield line.rstrip("\n\r")
        return
    if isinstance(src, (bytes, bytearray)):
        text = _decode_bytes(bytes(src))
        for line in text.splitlines():
            yield line.rstrip("\n\r")
        return
    # Fallback: assume it's already an iterable of strings
    for line in src:  # type: ignore[assignment]
        yield str(line).rstrip("\n\r")


def fasta_to_dataframe(
    src: Union[str, os.PathLike, bytes, Iterable[str]],
    *,
    include_ids: bool = False,
    seq_col: str = "sequence",
    id_col: str = "id",
) -> pd.DataFrame:
    """Parse a FASTA input into a pandas DataFrame.

    Parameters
    - src: Path, bytes, text, or iterable of lines containing FASTA records
    - include_ids: If True, include the sequence identifier column
    - seq_col: Name of the output column holding sequences
    - id_col: Name of the optional output column holding record ids

    Returns
    - pd.DataFrame with at least one column: `seq_col` containing sequences

    Notes
    - Collapses multiline sequences into single strings
    - Uses the first token after '>' as the id (up to first whitespace)
    """
    sequences: List[str] = []
    ids: List[Optional[str]] = []

    current_id: Optional[str] = None
    current_seq_parts: List[str] = []

    def _flush():
        nonlocal current_id, current_seq_parts
        if current_seq_parts:
            seq = "".join(current_seq_parts).replace(" ", "").strip()
            if seq:
                sequences.append(seq)
                ids.append(current_id)
        current_id = None
        current_seq_parts = []

    for raw in _iter_lines(src):
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            # header line: flush previous record first
            _flush()
            header = line[1:].strip()
            # take id up to first whitespace, fallback to full header if no spaces
            current_id = header.split()[0] if header else ""
        else:
            # sequence line
            current_seq_parts.append(line)

    # flush trailing record
    _flush()

    if include_ids:
        df = pd.DataFrame({id_col: ids, seq_col: sequences})
    else:
        df = pd.DataFrame({seq_col: sequences})

    return df


def looks_like_fasta(data: Union[bytes, str]) -> bool:
    """Heuristic check if raw bytes look like FASTA contents."""
    text = _decode_bytes(data).lstrip()
    # FASTA should start with '>' on the first non-empty line
    for line in text.splitlines():
        if line.strip() == "":
            continue
        return line.startswith(">")
    return False
