from __future__ import annotations

import math
from os import PathLike
from pathlib import Path
import tomli
from typing import Iterable, NamedTuple, TypeAlias


class Particle(NamedTuple):
    """Tuple container to store information about one Particle in the simulation.

    Implements the string magic method to format it to print into xyz-file format.

    :param name: str
        name of the Atom or Atomtype
    :param x: float
        x coordinate of particle
    :param y: float
        y coordinate of particle
    :param z: float
        z coordinate of particle
    """
    name: str
    x: float
    y: float
    z: float

    def __str__(self):
        return '\t'.join(str(prop) for prop in self)


class Frame(NamedTuple):
    """Tuple container to store information about one Frame in a simulation.

    Implements the string magic method to format the frame into xyz-file format.

    :param n_particles: int
        number of particles in the frame
    :param comment: str
        comment about that frame / structure. No restrictions here.
    :param particles: list of Particles
        particles in this Frame
    """
    n_particles: int
    comment: str
    particles: list[Particle]

    def __str__(self):
        return '\n'.join([str(self.n_particles), self.comment,
                          *[str(p) for p in self.particles]])

Topology: TypeAlias = dict[str, list[list[float]]]

# Particle = namedtuple('Particle', ['name', 'x', 'y', 'z'])
# Frame = namedtuple('Frame', ['n_particles', 'comment', 'particles'])

def read_xyz(file: Iterable[str]) -> list[Frame]:
    """Read XYZ-file or list of strings into Frames.

    The XYZ-file can contain multiple frames or only one frame. In any case a list of
    Frames will be returned.

    :param file: Iterable that returns strings
        Filehandle or list of strings that contain the XYZ-file information.

    :returns: list of Frames

    :raises: TypeError
        if the first line of a Frame does not contain a number denoting the
        number of particles in that Frame.
    :raises: ValueError
        if the number of particles does not match the number specified in the
        first line of a Frame.
    """
    file = iter(file)
    trajectory: list[Frame] = []
    while True:
        try:
            header: str = next(file).strip()
            n_particles: int = int(header)
        except StopIteration:
            return trajectory
        except ValueError:
            raise TypeError('Expected a integer denoting the number of particles in that frame, '
                             f'received {header!r}.')
        try:
            comment: str = next(file).strip()

            particles: list[Particle] = []
            for _ in range(n_particles):
                name, x, y, z = next(file).strip().split()
                x, y, z = float(x), float(y), float(z)

                particle = Particle(
                        name=name,
                        x=x,
                        y=y,
                        z=z,
                    )
                particles.append(particle)

            frame = Frame(
                    n_particles=n_particles,
                    comment=comment,
                    particles=particles,
                )
            trajectory.append(frame)
        except StopIteration:
            raise ValueError('File ended early. '
                             'Number of particles does not match or comment line is missing')


def generate_topology(coordinates: list[Particle], parameters: dict[str, dict[str, float]]) -> Topology:
    """Generate Topology dictionary that matches to the coordinates of one Frame.

    The topology is only valid for Frames that have the very same order of Particles as
    the Particle list the topology has been generated from.

    :param coordinates: list of Particles
        Particles in one Frame to generate the topology for
    :param parameters: dictionary
        This dictionary should contains the Atomname as key and as value a
        dictionary that maps parameters to value.

    :returns: Topology
        Dictionary with the parameter as key and as values a symmetric 2D list.
        The list contains in position i, j the parameter p_ij.
        For different parameters this can be either a arithmetic or geometric mean.
    """
    t_eps: list[list[float]] = [[0. for _ in coordinates] for _ in coordinates]
    t_r = t_eps.copy()
    for i, particle_a in enumerate(coordinates):
       for j, particle_b in enumerate(coordinates[i:], start=i):
           # geometric mean for epsilon parameter of lennard jones law
           t_eps[i][j] = math.sqrt(
                   parameters[particle_a.name]['eps'] * parameters[particle_b.name]['eps']
                )
           t_eps[j][i] = t_eps[i][j]
           # arithmetic mean for equilibrium distance r_min
           t_r[i][j] = (parameters[particle_a.name]['r_min'] + parameters[particle_b.name]['r_min']) / 2
           t_r[j][i] = t_r[i][j]

    topology = {
            'r_min' : t_r,
            'eps': t_eps,
        }
    return topology


def load_start_structure(filename: str | PathLike,
                         parameter_file: str | PathLike = Path(__file__).parent / 'parameters.toml') -> tuple[Frame, Topology]:
    """Open and read data from input files and generate start frame and correspoding topology."""
    with open(filename, 'r') as cf:
        first_frame = read_xyz(cf)[0]
    with open(parameter_file, 'rb') as pf:
        parameters = tomli.load(pf)
    topology = generate_topology(coordinates=first_frame.particles, parameters=parameters)
    return first_frame, topology
