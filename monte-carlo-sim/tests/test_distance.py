import pytest

from monte_carlo_sim.handle_xyz import Particle
from monte_carlo_sim.distance import non_periodic_squared, periodic_squared


near = [Particle(name='test1', x=0, y=0, z=0),
        Particle(name='test2', x=1, y=1, z=1)
        ]
far = [Particle(name='test1', x=0, y=0, z=0),
       Particle(name='test2', x=11, y=11, z=11)
       ]
particles = near

@pytest.mark.parametrize('function, particles, expected',
                         [
                             (non_periodic_squared, near, 3),
                             (periodic_squared(2, 2), near, 3),
                             (non_periodic_squared, far, 363),
                             (periodic_squared(2, 2), far, 3),
                         ]
                        )
def test_normal_no_wrap(function, particles, expected):
    """Test specific values."""
    distances = function(particles)
    assert distances[0][1] == expected


@pytest.mark.parametrize('function',
                         [non_periodic_squared,
                          periodic_squared(1, 2)])
def test_symmetry(function):
    """Test the symmetry of the result 2D Matrix."""
    dist = function(particles)
    assert all(dist[i][j] == dist[j][i] for i in range(2) for j in range(2))


@pytest.mark.parametrize('function',
                         [non_periodic_squared,
                          periodic_squared(10, 2)])
def test_diagonal(function):
    """Test the diagonal of the result. Distances to oneself should be 0."""
    dist = function(particles)
    assert all(dist[i][i] == 0 for i in range(2))
