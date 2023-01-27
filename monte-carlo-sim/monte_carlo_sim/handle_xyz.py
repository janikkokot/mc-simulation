"""File Operations and Datastructures"""
from __future__ import annotations

import math
from os import PathLike
from pathlib import Path
import tomli
from typing import Iterable, NamedTuple, TypeAlias

_base_path = Path(__file__).parent


class Particle(NamedTuple):
    """Tuple container to store information about one Particle in the simulation.

    The Particle tuple impolements the string magic method to format the tuple
    to match the XYZ-file format for Particles.
    In combination with Frames, this makes writing XYZ-files more intuitiv.

    Args:
        name (str): name of the Atom
            Name of the Atom may either be the Element Symbol or the Atomtype
        x (float): X coordinate of Atom in Angstroms
        y (float): Y coordinate of Atom in Angstroms
        z (float): Z coordinate of Atom in Angstroms

    Returns:
        tuple: containing the provided arguments in the order specified above.

    Examples:
        >>> hydrogen = Particle(name='H', x=0., y=0., z=0.)
        >>> hydrogen
        Particle(name='H', x=0.0, y=0.0, z=0.0)
        >>> str(hydrogen).split()
        ['H', '0.0', '0.0', '0.0']
        >>> hydrogen.name
        'H'
        >>> _, *coords = hydrogen
        >>> coords
        [0.0, 0.0, 0.0]
    """
    name: str
    x: float
    y: float
    z: float

    def __str__(self):
        return '\t'.join(str(prop) for prop in self)


class Frame(NamedTuple):
    """Tuple container to store information about one structure from XYZ-Format.

    A Frame is just one structure / configuration. Multiple of these Frames,
    where the number and order of Particles stays the same then form a
    trajectory.

    The Frame Tuple implements the string magic method making writing the frame
    into XYZ-file format more intuitiv.

    Args:
        n_particles (int): number of particles in the Frame / structure
        comment (str): comment about that Frame / structure
        particles (list[Particle]): individual particles in this Frame / structure

    Returns:
        tuple: containing all information associated with one structure.
            The order of the tuple elements is as specified above.

    Examples:
        >>> coords = [Particle('O', 0., 0., 0.),
        ...           Particle('H', 0.758602, 0., 0.504284),
        ...           Particle('H', 0.758602, 0., -0.504284),]
        >>> water = Frame(n_particles=3, comment='Water', particles=coords)
        >>> water.n_particles
        3
        >>> water.comment
        'Water'
        >>> len(water.particles)
        3
        >>> print(water)
        3
        Water
        ...
    """
    n_particles: int
    comment: str
    particles: list[Particle]

    def __str__(self):
        return '\n'.join([str(self.n_particles), self.comment,
                          *[str(p) for p in self.particles]])


# Particle = namedtuple('Particle', ['name', 'x', 'y', 'z'])
# Frame = namedtuple('Frame', ['n_particles', 'comment', 'particles'])
Topology: TypeAlias = dict[str, list[list[float]]]
"""dictionary, Topology type that matches to a given list of Particles.

The dictionary maps strings denoting the parameter name to a 2D list.
Entry $ij$ is the combined parameter from particles $i$ and $j$.
"""


def read_xyz(file: Iterable[str]) -> list[Frame]:
    """Read XYZ-file or list of strings into a list of Frames.

    The XYZ-file can contain multiple frames or only one frame.

    Args:
        file: contains the XYZ-file information.

    Returns:
        list[Frames]: all XYZ-file information stored in Frames.

    Raises:
        TypeError: If the first line of a Frame does not contain a number
            denoting the number of particles in that Frame.
        ValueError: If the number of particles does not match the number
            specified in the first line of a Frame.

    Examples:
        >>> file = ['  3 ', 'Angstrom ', '  O  0.000000  0.000000  0.000000',
        ...         '  H  0.758602  0.000000  0.504284',
        ...         '  H  0.758602  0.000000  -0.504284']
        >>> frames = read_xyz(file)
        >>> type(frames)
        <class 'list'>
        >>> len(frames)
        1
        >>> water = frames[0]
        >>> water.comment
        'Angstrom'
    """
    file = iter(file)
    trajectory: list[Frame] = []
    while True:
        try:
            header: str = next(file).strip()
        except StopIteration:
            return trajectory
        try:
            n_particles: int = int(header)
        except ValueError:
            raise TypeError('Expected a integer denoting the number of '
                            f'particles in that frame, received {header!r}.')
        try:
            comment: str = next(file).strip()

            particles: list[Particle] = []
            for _ in range(n_particles):
                name, _x, _y, _z = next(file).strip().split()
                x, y, z = float(_x), float(_y), float(_z)

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
            raise ValueError('File ended early. Number of particles does not '
                             'match or comment line is missing')


def generate_topology(coordinates: list[Particle],
                      parameters: dict[str, dict[str, float]],
                      ) -> Topology:
    """Generate a Topology dictionary for a given Particle list.

    The topology is only valid for Frames that have the very same order of
    Particles as the Particle list the topology has been generated from.

    Args:
        coordinates: Particles to generate the topology for
        parameters: nested dictionary containing parameters for atomtypes
            The mapping schematically is as follows:
                Atomtype -> parameter_name -> value

    Returns: Topology
        Dictionary with the parameter as key and as values a symmetric 2D list.
        The list contains in position $i$, $j$ the parameter $p_{ij}$.
        Parameter $p_{ij}$ can be either a arithmetic or geometric mean of
        $p_i$ and $p_j$.

    Examples:
        >>> a = Particle('a', 0, 0, 0)
        >>> b = Particle('b', 0, 0, 0)
        >>> parameters = {'a': {'eps': 2, 'r_min': 2},
        ...               'b': {'eps': 50, 'r_min': 4}}
        >>> top = generate_topology([a, b], parameters)
        >>> top['eps']
        [[2.0, 10.0], [10.0, 50.0]]
        >>> top['r_min']
        [[2.0, 3.0], [3.0, 4.0]]

    """
    t_eps: list[list[float]] = [[0. for _ in coordinates] for _ in coordinates]
    t_r: list[list[float]] = [[0. for _ in coordinates] for _ in coordinates]
    for i, particle_a in enumerate(coordinates):
        for j, particle_b in enumerate(coordinates[i:], start=i):
            _a = parameters[particle_a.name]
            _b = parameters[particle_b.name]
            # geometric mean for epsilon
            t_eps[i][j] = math.sqrt(
                    _a['eps'] * _b['eps']
                    )
            t_eps[j][i] = t_eps[i][j]
            # arithmetic mean for epsilon
            t_r[i][j] = (_a['r_min'] + _b['r_min']) / 2
            t_r[j][i] = t_r[i][j]

    topology = {
            'r_min': t_r,
            'eps': t_eps,
        }
    return topology


def load_start_structure(
        xyz_file: str | PathLike,
        parameter_file: str | PathLike = _base_path / 'parameters.toml',
        ) -> tuple[list[Particle], Topology]:
    """Read XYZ File and generate a start Frame with a correspoding topology.

    Args:
        xyz_file: file in XYZ format
            contains coordinate and atomtype information
        parameter (optional): parameter file in toml format
            has the general format of
                [Atomtype]
                parameter = value
            if not provided, use the default parameter file `parameters.toml`.

    Returns: list[Particle], Topology
        list with all particles and a matching topology.
    """
    with open(xyz_file, 'r') as coordinate_file:
        first_frame = read_xyz(coordinate_file)[0]
    with open(parameter_file, 'rb') as pf:
        parameters = tomli.load(pf)
    topology = generate_topology(
            coordinates=first_frame.particles,
            parameters=parameters,
            )
    return first_frame.particles, topology
