"""
Unit tests for the synthetic GGM data generator.
"""

import numpy as np
import pytest

from nodis.simulate.generator import generate, _precision_from_adjacency


TOPOLOGIES = ["hub", "scale-free", "cluster", "random"]


# ---------------------------------------------------------------------------
# Data shape and type
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_output_shape(topology):
    data = generate(n=60, p=15, topology=topology, seed=0)
    assert data.X.shape == (60, 15)
    assert data.Theta.shape == (15, 15)
    assert data.Omega.shape == (15, 15)


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_metadata(topology):
    data = generate(n=60, p=15, topology=topology, seed=42)
    assert data.n == 60
    assert data.p == 15
    assert data.topology == topology
    assert data.seed == 42


# ---------------------------------------------------------------------------
# Precision matrix properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_precision_positive_definite(topology):
    data = generate(n=100, p=20, topology=topology, seed=1)
    eigs = np.linalg.eigvalsh(data.Theta)
    assert eigs.min() > 0, (
        f"Precision matrix for '{topology}' is not positive definite "
        f"(min eigenvalue = {eigs.min():.6f})."
    )


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_precision_symmetric(topology):
    data = generate(n=60, p=15, topology=topology, seed=2)
    np.testing.assert_array_almost_equal(data.Theta, data.Theta.T, decimal=12)


# ---------------------------------------------------------------------------
# Adjacency properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_adjacency_binary(topology):
    data = generate(n=60, p=15, topology=topology, seed=3)
    assert set(np.unique(data.Omega)).issubset({0, 1})


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_adjacency_symmetric(topology):
    data = generate(n=60, p=15, topology=topology, seed=4)
    np.testing.assert_array_equal(data.Omega, data.Omega.T)


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_adjacency_no_self_loops(topology):
    data = generate(n=60, p=15, topology=topology, seed=5)
    np.testing.assert_array_equal(np.diag(data.Omega), np.zeros(15, dtype=int))


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_seed_reproducibility():
    data1 = generate(n=80, p=20, topology="hub", seed=99)
    data2 = generate(n=80, p=20, topology="hub", seed=99)
    np.testing.assert_array_equal(data1.X, data2.X)
    np.testing.assert_array_equal(data1.Omega, data2.Omega)


def test_different_seeds_differ():
    data1 = generate(n=80, p=20, topology="hub", seed=1)
    data2 = generate(n=80, p=20, topology="hub", seed=2)
    assert not np.array_equal(data1.X, data2.X)


# ---------------------------------------------------------------------------
# Edge case: unknown topology
# ---------------------------------------------------------------------------

def test_unknown_topology_raises():
    with pytest.raises(ValueError, match="Unknown topology"):
        generate(n=50, p=10, topology="butterfly", seed=0)


# ---------------------------------------------------------------------------
# Precision builder
# ---------------------------------------------------------------------------

def test_precision_from_adjacency_pd():
    rng = np.random.default_rng(0)
    A = (rng.random((15, 15)) < 0.1).astype(int)
    A = np.triu(A, 1)
    A = A + A.T
    np.fill_diagonal(A, 0)
    Theta = _precision_from_adjacency(A)
    eigs = np.linalg.eigvalsh(Theta)
    assert eigs.min() > 0
