"""
Tests for the seed-offset mechanism used in run_synthetic.py.

Verifies that:
- Different seed_offsets produce different data
- Same seed_offset always produces identical data (reproducibility)
- Seed ranges across offsets/reps/topologies are non-overlapping
"""

import pytest
import numpy as np

from nodis.simulate.generator import generate


def _compute_seed(seed_offset: int, rep: int, topology: str) -> int:
    """Mirrors the seed formula in benchmarks/run_synthetic.py."""
    return seed_offset * 100_000 + rep * 1000 + hash(topology) % 1000


TOPOLOGIES = ["hub", "scale-free", "cluster", "random"]


class TestSeedOffsetMechanism:
    def test_different_offsets_produce_different_data(self):
        """seed_offset=0 and seed_offset=1 must yield different samples."""
        seed0 = _compute_seed(seed_offset=0, rep=0, topology="hub")
        seed1 = _compute_seed(seed_offset=1, rep=0, topology="hub")
        d0 = generate(n=30, p=10, topology="hub", seed=seed0)
        d1 = generate(n=30, p=10, topology="hub", seed=seed1)
        assert not np.allclose(d0.X, d1.X), (
            "Different seed_offsets must produce different data"
        )

    def test_same_offset_is_reproducible(self):
        """Same seed_offset/rep/topology always produces identical data."""
        seed = _compute_seed(seed_offset=2, rep=7, topology="cluster")
        d_a = generate(n=30, p=10, topology="cluster", seed=seed)
        d_b = generate(n=30, p=10, topology="cluster", seed=seed)
        assert np.allclose(d_a.X, d_b.X), "Identical seeds must produce identical data"

    def test_no_seed_collisions_across_offsets_and_reps(self):
        """Seeds for (offset, rep, topology) combinations must not collide."""
        seeds = set()
        for offset in range(3):
            for rep in range(50):
                for topo in TOPOLOGIES:
                    s = _compute_seed(offset, rep, topo)
                    assert s not in seeds, (
                        f"Seed collision at offset={offset}, rep={rep}, topo={topo}"
                    )
                    seeds.add(s)

    @pytest.mark.parametrize("topology", TOPOLOGIES)
    def test_offset_spacing_prevents_overlap(self, topology):
        """seed_offset steps of 100_000 must keep offsets non-adjacent regardless of rep."""
        max_rep_contribution = 49 * 1000 + max(abs(hash(t)) % 1000 for t in TOPOLOGIES)
        assert max_rep_contribution < 100_000, (
            "rep*1000 + topo_hash can overflow into the next offset block"
        )
