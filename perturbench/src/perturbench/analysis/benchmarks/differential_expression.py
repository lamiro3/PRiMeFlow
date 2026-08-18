"""Backward compatibility import for rank genes functions.

This module re-exports functions from _rank_genes_helpers to maintain
backward compatibility while avoiding code duplication.
"""

from ._rank_genes_helpers import (
    _RankGenesControlVar,
    rank_genes_groups_control_var,
)

__all__ = ["_RankGenesControlVar", "rank_genes_groups_control_var"]
