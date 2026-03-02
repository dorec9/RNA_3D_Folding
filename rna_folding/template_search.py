"""
Template-Based Modeling (TBM) pipeline for RNA 3D structure prediction.

Implements the 1st-place approach from the Stanford RNA 3D Folding competition:
- MMseqs2 easy-search with ``--search-type 3`` (nucleotide mode)
- Top-5 template extraction producing one prediction per ranked template
- C1' coordinate extraction from mmCIF files via the DasLab alignment protocol
- Linear interpolation for gap filling (FRABASE-style fragment insertion is a
  drop-in replacement if the fragment library is available)

References:
    DasLab/create_templates: github.com/DasLab/create_templates
    1st place write-up: jaejohn (G. John Rao), Kaggle discussion thread
"""

import os
import subprocess
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from rna_folding.coordinate_utils import (
    extract_c1_prime_from_cif,
    build_coordinate_array,
    fill_gaps_linear,
)

logger = logging.getLogger(__name__)

# E-value threshold below which a template is considered a strong hit.
# Targets with at least one hit below this threshold can rely on TBM alone.
STRONG_HIT_EVALUE = 1e-5


class TemplateSearcher:
    """
    TBM pipeline: MMseqs2 search → coordinate extraction → gap filling.

    Usage::

        searcher = TemplateSearcher(
            pdb_db_path="/data/pdb_seqres_NA",
            cif_dir="/data/pdb_mmcif",
        )
        candidates = searcher.get_tbm_candidates(sequence, top_k=5)
    """

    # MMseqs2 output columns (--format-output value)
    _MMSEQS_COLS = [
        "query", "target", "fident", "alnlen", "mismatch", "gapopen",
        "qstart", "qend", "tstart", "tend", "evalue", "bits", "qaln", "taln",
    ]
    _MMSEQS_FORMAT = ",".join(_MMSEQS_COLS)

    def __init__(
        self,
        pdb_db_path: str,
        cif_dir: str = "",
        mmseqs_bin: str = "mmseqs",
        tmp_dir: str = "/tmp/mmseqs_rna",
        max_seqs: int = 1000,
    ) -> None:
        """
        Args:
            pdb_db_path: Path prefix of the MMseqs2 database built from
                         PDB RNA sequences (e.g. ``/data/pdb_seqres_NA``).
                         Build with::

                             mmseqs createdb pdb_seqres_NA.fasta pdb_seqres_NA \\
                                 --dbtype 2

            cif_dir: Directory containing mmCIF files named ``<pdb_id>.cif``
                     or ``<pdb_id>.cif.gz``.  If empty, coordinate extraction
                     is skipped and only simulated (zero) coordinates are used.
            mmseqs_bin: Path or name of the MMseqs2 binary.
            tmp_dir: Temporary directory for MMseqs2 intermediate files.
            max_seqs: Maximum number of sequences returned by the search
                      (passed to MMseqs2 as ``--max-seqs``).
        """
        self.pdb_db_path = pdb_db_path
        self.cif_dir = cif_dir
        self.mmseqs_bin = mmseqs_bin
        self.tmp_dir = tmp_dir
        self.max_seqs = max_seqs
        os.makedirs(self.tmp_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # MMseqs2 search
    # ------------------------------------------------------------------

    def run_mmseqs_search(self, sequence: str, target_id: str = "query") -> pd.DataFrame:
        """
        Search *sequence* against the PDB RNA database with MMseqs2.

        Uses ``--search-type 3`` (nucleotide) as established by the 1st-place
        competition solution.  Results are sorted ascending by e-value.

        Args:
            sequence: Target RNA sequence (ACGU alphabet).
            target_id: Identifier written into the query FASTA header.

        Returns:
            DataFrame with columns matching ``_MMSEQS_COLS``, sorted by
            ascending e-value.  Empty DataFrame on failure or no hits.
        """
        fasta_path = os.path.join(self.tmp_dir, f"{target_id}_query.fasta")
        result_path = os.path.join(self.tmp_dir, f"{target_id}_results.m8")

        with open(fasta_path, "w") as fh:
            fh.write(f">{target_id}\n{sequence}\n")

        cmd = [
            self.mmseqs_bin, "easy-search",
            fasta_path,
            self.pdb_db_path,
            result_path,
            self.tmp_dir,
            "--search-type", "3",          # nucleotide mode — key for RNA TBM
            "--format-output", self._MMSEQS_FORMAT,
            "--max-seqs", str(self.max_seqs),
            "-v", "1",                     # minimal stdout noise
        ]

        logger.debug("Running MMseqs2: %s", " ".join(cmd))
        try:
            subprocess.run(
                cmd, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as exc:
            logger.warning("MMseqs2 failed: %s", exc.stderr.decode())
            return pd.DataFrame()
        except FileNotFoundError:
            logger.warning(
                "MMseqs2 binary not found at '%s'. Skipping TBM.", self.mmseqs_bin
            )
            return pd.DataFrame()

        if not os.path.exists(result_path) or os.path.getsize(result_path) == 0:
            logger.info("No MMseqs2 hits for target %s.", target_id)
            return pd.DataFrame()

        df = pd.read_csv(result_path, sep="\t", names=self._MMSEQS_COLS)
        df = df.sort_values(["evalue", "bits"], ascending=[True, False]).reset_index(drop=True)
        return df

    def has_strong_template(self, hits: pd.DataFrame) -> bool:
        """
        Return True if the top hit has e-value below ``STRONG_HIT_EVALUE``.

        When True, TBM predictions alone may achieve near-optimal accuracy and
        deep learning inference can be deprioritised to save GPU time.
        """
        if hits.empty:
            return False
        return float(hits.iloc[0]["evalue"]) < STRONG_HIT_EVALUE

    # ------------------------------------------------------------------
    # Coordinate extraction
    # ------------------------------------------------------------------

    def _find_cif(self, pdb_id: str) -> Optional[str]:
        """Locate a mmCIF file by PDB ID (case-insensitive, with/without .gz)."""
        if not self.cif_dir:
            return None
        for suffix in ["", ".gz"]:
            for name in [pdb_id.lower(), pdb_id.upper()]:
                path = os.path.join(self.cif_dir, f"{name}.cif{suffix}")
                if os.path.exists(path):
                    return path
        return None

    def extract_template_coords(
        self, sequence: str, hit: pd.Series
    ) -> np.ndarray:
        """
        Map C1' coordinates from a template hit onto the query sequence.

        Parses the alignment strings (``qaln`` / ``taln``) to place template
        residue coordinates at the correct query positions.  Gaps are left as
        NaN and filled by the caller.

        Args:
            sequence: Full target RNA sequence.
            hit: A single row from the MMseqs2 result DataFrame.

        Returns:
            ``(len(sequence), 3)`` float32 array; NaN where not covered.
        """
        seq_len = len(sequence)

        # Parse target identifier: MMseqs2 writes "<pdb_id>_<chain>" or "<pdb_id>"
        target_field = str(hit["target"])
        parts = target_field.split("_")
        pdb_id = parts[0]
        chain_id = parts[1] if len(parts) > 1 else None

        cif_path = self._find_cif(pdb_id)
        if cif_path is None:
            # No CIF available — return zero array as a last resort
            logger.debug("CIF not found for %s; using zero coordinates.", pdb_id)
            return np.zeros((seq_len, 3), dtype=np.float32)

        coord_dict = extract_c1_prime_from_cif(cif_path, chain_id=chain_id)
        if not coord_dict:
            logger.debug("No C1' atoms parsed from %s chain %s.", pdb_id, chain_id)
            return np.zeros((seq_len, 3), dtype=np.float32)

        # MMseqs2 uses 1-based inclusive indices for both query and target
        q_start = int(hit["qstart"]) - 1  # convert to 0-based
        t_start = int(hit["tstart"])      # keep 1-based (matches coord_dict keys)

        coords = build_coordinate_array(
            seq_len=seq_len,
            coord_dict=coord_dict,
            q_start=q_start,
            q_end=int(hit["qend"]),
            t_start=t_start,
            t_end=int(hit["tend"]),
            qaln=str(hit["qaln"]),
            taln=str(hit["taln"]),
        )
        return coords

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tbm_candidates(
        self, sequence: str, top_k: int = 5, target_id: str = "query"
    ) -> List[np.ndarray]:
        """
        Run the full TBM pipeline and return up to *top_k* candidate structures.

        Each candidate corresponds to one ranked template hit, so the best
        template produces candidate 0, the second-best produces candidate 1,
        and so on.  This directly implements the "top-1 through top-5 templates
        each generate one of the five required predictions" strategy from the
        1st-place solution.

        Args:
            sequence: Target RNA sequence.
            top_k: Maximum number of candidates to return (≤ 5 for competition).
            target_id: Identifier used in temporary FASTA and result file names.

        Returns:
            List of ``(len(sequence), 3)`` float32 arrays, length ≤ top_k.
            Empty list if no templates are found.
        """
        hits = self.run_mmseqs_search(sequence, target_id=target_id)
        if hits.empty:
            return []

        candidates: List[np.ndarray] = []
        for rank in range(min(top_k, len(hits))):
            hit = hits.iloc[rank]
            raw_coords = self.extract_template_coords(sequence, hit)
            filled_coords = fill_gaps_linear(raw_coords)
            candidates.append(filled_coords)

        return candidates

    def get_tbm_candidates_with_evalue(
        self, sequence: str, top_k: int = 5, target_id: str = "query"
    ) -> List[Dict]:
        """
        Like ``get_tbm_candidates`` but also returns e-values for each hit.

        Returns:
            List of dicts with keys ``coords`` (np.ndarray) and ``evalue`` (float).
        """
        hits = self.run_mmseqs_search(sequence, target_id=target_id)
        if hits.empty:
            return []

        results = []
        for rank in range(min(top_k, len(hits))):
            hit = hits.iloc[rank]
            raw_coords = self.extract_template_coords(sequence, hit)
            filled_coords = fill_gaps_linear(raw_coords)
            results.append({
                "coords": filled_coords,
                "evalue": float(hit["evalue"]),
                "fident": float(hit["fident"]),
                "target": str(hit["target"]),
            })

        return results
