from typing import TypeAlias, Callable
from monte_carlo_sim import handle_xyz as xyz


DistanceFunction: TypeAlias = Callable[
        [list[xyz.Particle]],
        xyz.Matrix2D,
        ]


def periodic_squared(density: float,
                     n_particles: int,
                     ) -> DistanceFunction:
    """Return function to calculate periodic calculated squared distances.

    The cubic box size is calculated from the desired density and the
    particle count.

    Args:
        density: number of particles per cube nanometer
        n_particles: number of particles in the cell

    Returns: Function calculating periodic pairwise squared distances.

    Examples:
        >>> points = [xyz.Particle('x', 0., 0., -4.),
        ...           xyz.Particle('x', 0., 0., 4.),]
        >>> f = periodic_squared(density=2, n_particles=2)
        >>> f(points)
        [[0.0, 4.0], [4.0, 0.0]]
    """
    box_side = (n_particles / density) ** (1/3) * 10  # to Angstrom

    def distance(a, b):
        """Return distance between two points in a periodic cubic cell."""
        dist = b-a
        return dist - box_side * round(dist / box_side)

    def distance_squared(particles: list[xyz.Particle]):
        """Calculate pairwise squared distances.

        The lennard jones function expects squared distances, this way the
        calculation of the square and the square root for each pair can be
        avoided.

        Args:
            particles: Particles between the distances will be calculated

        Returns: Matrix2D, pairwise distances
        """
        dist_squared = [[0. for _ in particles] for _ in particles]
        for i, a in enumerate(particles[:-1]):
            for j, b in enumerate(particles[i:], start=i):
                r2 = (distance(a.x, b.x)**2 +
                      distance(a.y, b.y)**2 +
                      distance(a.z, b.z)**2)
                dist_squared[i][j] = dist_squared[j][i] = r2
        return dist_squared
    return distance_squared


def non_periodic_squared(
        particles: list[xyz.Particle]
        ) -> xyz.Matrix2D:
    """Calculate pairwise squared distances.

    The lennard jones function expects squared distances, this way the
    calculation of the square and the square root for each pair can be avoided.

    Args:
        particles: Particles between the distances will be calculated

    Returns: Matrix2D, pairwise distances

    Examples:
        >>> points = [xyz.Particle('x', 0., 0., -4.),
        ...           xyz.Particle('x', 0., 0., 4.),]
        >>> non_periodic_squared(points)
        [[0.0, 64.0], [64.0, 0.0]]
    """
    dist_squared = [[0. for _ in particles] for _ in particles]
    for i, a in enumerate(particles[:-1]):
        for j, b in enumerate(particles[i:], start=i):
            r2 = (b.x - a.x)**2 + (b.y - a.y)**2 + (b.z - a.z)**2
            dist_squared[i][j] = dist_squared[j][i] = r2
    return dist_squared
