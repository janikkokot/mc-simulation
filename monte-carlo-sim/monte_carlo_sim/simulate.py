"""Simulation and energy calculations"""
from __future__ import annotations

import math
import random
from typing import Callable
import warnings

from monte_carlo_sim import handle_xyz
from monte_carlo_sim import distance as dist


BOLTZMANN = 1.
"""float, Boltzman constant normally 1.380649e-23 J/K. But epsilon parameter
energy depth is given in energy/Boltzmann.
"""


def monte_carlo(
        start_coordinates: list[handle_xyz.Particle],
        topology: handle_xyz.Topology,
        steps: int,
        temperature: float = 300.,
        distance_squared: dist.DistanceFunction = dist.non_periodic_squared,
        get_step_size: Callable[[bool], float] = lambda _: 0.01,
        ) -> list[handle_xyz.Frame]:
    """Run a Monte-Carlo simulation.

    Args:
        start_coordinates: Coordinates from which the simulation is started
        topology: Topology with parameter information
            matches the start coordinates
        steps: number of iteration
        temperature: Temperature in Kelvin
        distance_squared: Callable returning the squared pairwise distances
            The distances are in Angstrom.
            This setup allows to switch easily between a periodic and a non-
            periodic simulation setup.
        get_step_size: Callable returning the step size in Angstrom
            The Callable takes a bool as argument, should be the Metropolis
            criterion results. This allows to have more complex step_sizes,
            e.g. an addaptive step size adjustment.

    Returns: list[Frame], Trajectory of the generated ensemble.

    Examples:
        >>> points = [handle_xyz.Particle('x', 0, 0, 0),
        ...           handle_xyz.Particle('x', 0, 0, 2.5),]
        >>> params = {'x' : {'r_min': 2, 'eps': 1}}
        >>> top = handle_xyz.generate_topology(points, params)
        >>> traj = monte_carlo(start_coordinates=points,
        ...                 topology=top,
        ...                 steps=10, # usually in the range of millions
        ...                 temperature=300, #K
        ...                 )
        >>> type(traj)
        <class 'list'>
        >>> len(traj)
        10
        >>> [a == b for a, b in zip(traj[0].particles, points)]
        [True, True]
        >>> for frame in traj:
        ...     for particle in frame.particles:
        ...         assert particle.name == 'x'
    """
    n_particles = len(start_coordinates)
    step_size = get_step_size(True)

    distances = distance_squared(start_coordinates)
    energy = get_conformation_energy(distances, topology)
    trajectory = [
            handle_xyz.Frame(
                n_particles=n_particles,
                comment=f'step 0, {temperature}, {energy}, {step_size}',
                particles=start_coordinates,
                )
        ]
    try:
        while len(trajectory) < steps:
            trial_structure: list[handle_xyz.Particle] = perturbe_structure(
                    particles=trajectory[-1].particles,
                    step_size=step_size,
                    )
            distances = distance_squared(trial_structure)
            trial_energy: float = get_conformation_energy(
                    distances=distances,
                    topology=topology
                    )

            accepted: bool = metropolis_criterion(
                    old_energy=energy,
                    new_energy=trial_energy,
                    temperature=temperature,
                    )
            step_size = get_step_size(accepted)
            if accepted:
                energy = trial_energy
                n_frame = len(trajectory)
                new_frame = handle_xyz.Frame(
                        n_particles=n_particles,
                        comment=(f'step {n_frame}, {temperature}, '
                                 f'{energy}, {step_size}'),
                        particles=trial_structure,
                    )
                trajectory.append(new_frame)
    except BaseException as err:
        warnings.warn(f'Something unexpected happened: {err!r}')
    return trajectory


def perturbe_structure(particles: list[handle_xyz.Particle],
                       step_size: float,
                       ) -> list[handle_xyz.Particle]:
    """Translate a random particle in a random direction.

    The magnitude of the step is determined by the step_size.

    Args:
        particles: initial configuration of the system
        step_size: magnitude of the random change

    Returns: new conformation


    Examples:
        >>> points = [handle_xyz.Particle('x', 0., 0., 0.),
        ...           handle_xyz.Particle('x', 0., 0., 2.),]
        >>> a, b = points
        >>> c, d = perturbe_structure(points, 0.5)
        >>> (a != c and b == d) or (a == c and b != d)
        True
    """
    particles = particles.copy()

    x = random.random() - 0.5
    y = random.random() - 0.5
    z = random.random() - 0.5
    length = math.sqrt(x*x + y*y + z*z) / step_size

    idx = random.randrange(len(particles))
    chosen = particles[idx]
    perturbed = handle_xyz.Particle(
            name=chosen.name,
            x=chosen.x+x/length,
            y=chosen.y+y/length,
            z=chosen.z+z/length,
        )
    particles[idx] = perturbed
    return particles


def plain_step_size(step_size: float):
    """Returns function that returns the step size no matter what.

    Args:
        step_size: initial step size in Angstrom

    Returns: Function taking a boolean and always return the same step size.

    Examples:
        >>> f = plain_step_size(0.5)
        >>> f(False)
        0.5
        >>> f(True)
        0.5
    """
    return lambda _: step_size


def addaptive_step_size(step_size: float,
                        target_ratio: float,
                        update_frequency: float,
                        ) -> Callable[[bool], float]:
    """Returns function that returns the step size.

    The step size is addapted based on the metropolis acceptance ratio of
    previous steps.

    Args:
        step_size: initial step size in Angstrom
        target_ratio: ratio of accepted to declined MC steps targeted
        update_frequency: number of steps until step size is adjusted

    Returns: Function taking the metropolis result and returning the step size.

    Examples:
        >>> f = addaptive_step_size(0.5, 1, 2)
        >>> f(False)
        0.5
        >>> f(True)
        0.4
    """

    decisions: list[bool] = []
    step = {'size': step_size}

    def variable_step_size(decision: bool) -> float:
        decisions.append(decision)
        if len(decisions) >= update_frequency:
            ratio = math.sqrt(
                    sum(decisions) / (target_ratio * update_frequency)
                    )
            _lower, ratio, _upper = sorted([0.8, ratio, 1.2])
            step['size'] = min(step['size'] * ratio, 1)
            decisions.clear()
        return step['size']
    return variable_step_size


def metropolis_criterion(new_energy: float,
                         old_energy: float,
                         temperature: float = 300.,
                         ) -> bool:
    """Acceptance criterion of a Monte-Carlo step.

    If the energy of the new conformation is lower than the previous
    conformation, then the new conformation is always accepted.
    Otherwise a Boltzmann-factor is compared to a random number. If the random
    number is lower than the Boltzmann-factor, the higher energy conformation
    is accepted into the ensemble.

    Examples:
        >>> metropolis_criterion(new_energy=0,
        ...                      old_energy=1,
        ...                      temperature=300)
        True
        >>> metropolis_criterion(new_energy=1,
        ...                      old_energy=0,
        ...                      temperature=0+1e-10)
        False
    """
    de = new_energy - old_energy
    return (de <= 0) or \
           (random.random() < math.exp(-de / (BOLTZMANN * temperature)))


def lennard_jones(r_squared: float, r_min: float, eps: float) -> float:
    r"""Lennard-Jones law that expects the squared distance.

    This way the computation of the squareroot can be avoided.

    $V_{LJ} = \eps \cdot \left((\frac{r_{min}}{r_{actual}})^{12} - 2 \cdot
    (\frac{r_{min}}{r_{actual}})^6\right)$

    Args:
        r_squared: distance between the two interacting Particles squared
        eps: depth of the minimum

    Returns: float, Lennard-Jones Potential

    Examples:
        >>> r = 2
        >>> lennard_jones(r_squared=r**2, r_min=2, eps=1)
        -1.0
        >>> eps = 1e-7
        >>> lennard_jones((r-eps)**2, 2, 1) > lennard_jones(r**2, 2, 1)
        True
        >>> lennard_jones(r**2, 2, 1) < lennard_jones((r+eps)**2, 2, 1)
        True
    """

    r6 = r_min**6 / r_squared**3
    return eps * r6 * (r6 - 2)


def get_conformation_energy(distances: handle_xyz.Matrix2D,
                            topology: handle_xyz.Topology,
                            ) -> float:
    """Calculate energy associated with a conformation.

    Args:
        distances: pairwise distances
            All atoms in the conformation to be evaluated
        topology: Topology associated with that particular conformation

    Returns: float, Energy

    Examples:
        >>> points = [handle_xyz.Particle('x', 0, 0, 0),
        ...           handle_xyz.Particle('x', 0, 0, 2),]
        >>> distances = dist.non_periodic_squared(points)
        >>> params = {'x' : {'r_min': 2, 'eps': 1}}
        >>> top = handle_xyz.generate_topology(points, params)
        >>> get_conformation_energy(distances, top)
        -1.0
    """
    energy = 0.
    for i in range(len(distances)-1):
        for j in range(i+1, len(distances)):
            r2 = distances[i][j]
            try:
                energy += lennard_jones(
                        r_squared=r2,
                        r_min=topology['r_min'][i][j],
                        eps=topology['eps'][i][j],
                    )
            except ZeroDivisionError:
                continue
    return energy
