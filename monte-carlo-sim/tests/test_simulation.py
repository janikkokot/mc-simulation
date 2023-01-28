import pytest

from monte_carlo_sim.handle_xyz import Particle
from monte_carlo_sim.distance import non_periodic_squared, periodic_squared


particles = [Particle(name='test1', x=0, y=0, z=0),
             Particle(name='test2', x=1, y=1, z=1),
             ]


def test_value_non_periodic():
    distances = non_periodic_squared(particles)
    assert distances[0][1] == 3


@pytest.mark.parametrize('function',
                         [non_periodic_squared,
                          periodic_squared(2, 2)])
def test_symmetry(function):
    dist = function(particles)
    assert all(dist[i][j] == dist[j][i] for i in range(2) for j in range(2))


@pytest.mark.parametrize('function',
                         [non_periodic_squared,
                          periodic_squared(2, 2)])
def test_diagonal(function):
    dist = function(particles)
    assert all(dist[i][i] == 0 for i in range(2))
