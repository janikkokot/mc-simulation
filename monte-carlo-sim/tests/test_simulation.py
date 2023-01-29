import itertools
import pytest
from monte_carlo_sim import simulate
from monte_carlo_sim import handle_xyz as xyz
from monte_carlo_sim import distance as dist


def test_energy_calulation():
    top = {'r_min': [[float('nan'), 3], [3, float('nan')]],
           'eps': [[float('nan'), 10], [10, float('nan')]],}
    # diagonal entries not needed
    distances_squared = [[0, 9], [9, 0]]
    res = simulate.get_conformation_energy(distances_squared, top)
    assert res == -10


def test_energy_calulation_0division():
    """Can only happen when two particles are ontop of each other.
    In this case it is fair to crash."""

    top = {'r_min': [[float('nan'), 3, 3], [3, float('nan'), 3], [3, 3, float('nan')]],
           'eps': [[float('nan'), 10, 10], [10, float('nan'), 10], [10, 10, float('nan')]],}
    # diagonal entries not needed
    distances_squared = [[0, 9, 9], [9, 0, 0], [9, 0, 0]]
    with pytest.raises(ZeroDivisionError):
        simulate.get_conformation_energy(distances_squared, top)


@pytest.mark.parametrize('r,eps',
                         [(i, j) for i, j in itertools.combinations_with_replacement(range(1, 20, 3), 2)]
                         )
def test_lennard_jones_minimum(r, eps):
    assert simulate.lennard_jones(r**2, r, eps) == -eps
    dr = 1e-7
    assert simulate.lennard_jones((r-dr)**2, r, eps) > simulate.lennard_jones(r**2, r, eps)
    assert simulate.lennard_jones((r+dr)**2, r, eps) > simulate.lennard_jones(r**2, r, eps)


def test_lj_0division():
    with pytest.raises(ZeroDivisionError):
        simulate.lennard_jones(0, 1, 1)


def test_metropolis_acceptance():
    assert all(simulate.metropolis_criterion(0, 1, t) for t in range(0, 500, 50))


def test_metropolis_0K():
    for new, old in itertools.combinations_with_replacement(range(10), 2):
        assert simulate.metropolis_criterion(new, old, 0) == (new <= old)


def test_metropolis_decision():
    results = (simulate.metropolis_criterion(new, old, 300)
               for old, new in itertools.combinations(range(0, 1000, 10), 2))
    assert not all(results)


@pytest.mark.parametrize('steps, kwargs',
                         [(100, dict(
                             distance_squared=dist.non_periodic_squared,
                             get_step_size=simulate.plain_step_size(0.1),
                             )),
                          (100, dict(
                              distance_squared=dist.periodic_squared(2, 2),
                              get_step_size=simulate.plain_step_size(0.1),
                              )),
                          (100, dict(
                              distance_squared=dist.non_periodic_squared,
                              get_step_size=simulate.addaptive_step_size(0.1, 0.35, 20),
                              )),
                          (100, dict(
                              distance_squared=dist.periodic_squared(2, 2),
                              get_step_size=simulate.addaptive_step_size(0.1, 0.35, 20),
                              )),
                          ]
                         )
def test_monte_carlo_integration(steps, kwargs):
    _top = {'r_min': [[2, 2], [2, 2]],
           'eps': [[2, 2], [2, 2]]}
    _start = [xyz.Particle('1', 0, 0, 0),
             xyz.Particle('2', 1, -1, 4)]

    traj = simulate.monte_carlo(start_coordinates=_start,
                                topology=_top,
                                steps=steps,
                                temperature=300,
                                **kwargs
                                )
    assert sum(1 for _ in traj) == steps + 1
