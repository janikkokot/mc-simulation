"""Simulation, energy and distance calculations

TODO: refactor simulare function.
"""
from __future__ import annotations

import concurrent.futures as cf
import math
import random

import matplotlib.pyplot as plt  # type: ignore

from monte_carlo_sim import handle_xyz


BOLTZMANN = 1.380649e-23  # J/K
"""float, Boltzman constant in Joule per Kelvin."""


def simulate(start_coordinates: list[handle_xyz.Particle],
             topology: handle_xyz.Topology,
             steps: int,
             temperature: float = 300.,
             ):
    """Run a Monte-Carlo simulation.

    Args:
        start_coordinates: Coordinates from which the simulation is started
        topology: Topology with parameter information
            matches the start coordinates
        steps: number of iteration
        temperature: Temperature in Kelvin

    Returns: list[Frame], Trajectory of the generated ensemble.

    Examples:
        >>> points = [handle_xyz.Particle('x', 0, 0, 0),
        ...           handle_xyz.Particle('x', 0, 0, 2),]
        >>> params = {'x' : {'r_min': 2, 'eps': 1}}
        >>> top = handle_xyz.generate_topology(points, params)
        >>> traj = simulate(start_coordinates=points,
        ...                 topology=top,
        ...                 steps=10, # usually in the range of millions
        ...                 temperature=300, #K
        ...                 )
        >>> type(traj)
        <class 'list'>
        >>> [a == b for a, b in zip(traj[0].particles, points)]
        [True, True]
        >>> for frame in traj:
        ...     for particle in frame.particles:
        ...         assert particle.name == 'x'
    """
    first_frame = handle_xyz.Frame(
            n_particles=len(start_coordinates),
            comment='step 0',
            particles=start_coordinates,
        )
    trajectory = [first_frame]
    last_energy = get_conformation_energy(first_frame.particles, topology)

    for _ in range(steps):
        trial_structure = perturbe_structure(trajectory[-1].particles)
        trial_energy = get_conformation_energy(trial_structure, topology)

        if metropolis_criterion(
                old_energy=last_energy,
                new_energy=trial_energy,
                temperature=temperature,
                ):
            last_energy = trial_energy
            n_frame = len(trajectory)
            new_frame = handle_xyz.Frame(
                    n_particles=first_frame.n_particles,
                    comment=f'step {n_frame}',
                    particles=trial_structure,
                )
            trajectory.append(new_frame)
    return trajectory


def perturbe_structure(particles: list[handle_xyz.Particle],
                       step_length: float = 0.05,
                       ) -> list[handle_xyz.Particle]:
    """Translate a random particle in a random direction.

    The magnitude of the step is determined by the step_length.

    Args:
        particles: initial configuration of the system
        step_length: magnitude of the random change

    Returns: new conformation


    Examples:
        >>> points = [handle_xyz.Particle('x', 0., 0., 0.),
        ...           handle_xyz.Particle('x', 0., 0., 2.),]
        >>> a, b = points
        >>> c, d = perturbe_structure(points)
        >>> (a != c and b == d) or (a == c and b != d)
        True
    """
    particles = particles.copy()

    x = random.random() - 0.5
    y = random.random() - 0.5
    z = random.random() - 0.5
    length = math.sqrt(x*x + y*y + z*z) / step_length

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


def calculate_distance_squared(
        points: list[handle_xyz.Particle]
        ) -> list[list[float]]:
    """Calculate pairwise squared distances.

    The lennard jones function expects squared distances, this way the
    calculation of the square and the square root for each pair can be avoided.

    Examples:
        >>> points = [handle_xyz.Particle('x', 0., 0., 0.),
        ...           handle_xyz.Particle('x', 0., 0., 2.),]
        >>> calculate_distance_squared(points)
        [[0.0, 4.0], [4.0, 0.0]]
    """
    dist_squared = [[0. for _ in points] for _ in points]
    for i, a in enumerate(points[:-1]):
        for j, b in enumerate(points[i:], start=i):
            r2 = (b.x - a.x)**2 + (b.y - a.y)**2 + (b.z - a.z)**2
            dist_squared[i][j] = dist_squared[j][i] = r2
    return dist_squared


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


def get_conformation_energy(points: list[handle_xyz.Particle],
                            parameters: handle_xyz.Topology,
                            ) -> float:
    """Calculate energy associated with a conformation.

    Args:
        points: conformation to be evaluated
        parameters: Topology associated with that particular conformation

    Returns: float, Energy

    Examples:
        >>> points = [handle_xyz.Particle('x', 0, 0, 0),
        ...           handle_xyz.Particle('x', 0, 0, 2),]
        >>> params = {'x' : {'r_min': 2, 'eps': 1}}
        >>> top = handle_xyz.generate_topology(points, params)
        >>> get_conformation_energy(points, top)
        -1.0
    """
    distances = calculate_distance_squared(points)
    energy = 0.
    for i in range(len(distances)-1):
        for j in range(i+1, len(distances)):
            r2 = distances[i][j]
            try:
                energy += lennard_jones(
                        r_squared=r2,
                        r_min=parameters['r_min'][i][j],
                        eps=parameters['eps'][i][j],
                    )
            except ZeroDivisionError:
                continue
    return energy


def main():
    start, top = handle_xyz.load_start_structure('tests/start.xyz')
    with cf.ProcessPoolExecutor() as executor:
        fs = []
        for _ in range(6):
            fs.append(executor.submit(simulate, start, top, 100_000))

        for n, future in enumerate(cf.as_completed(fs)):
            traj = future.result()
            with open(f'out_{n}.xyz', 'w') as traj_file:
                traj_file.write('\n'.join(str(frame) for frame in traj))
            en = [get_conformation_energy(frame.particles, top)
                  for frame in traj]
            plt.plot(en, '.', label=f'run {n}')
    plt.show()


if __name__ == '__main__':
    main()
