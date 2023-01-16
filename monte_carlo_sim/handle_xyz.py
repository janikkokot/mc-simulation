from typing import Iterable, NamedTuple
#from collections import namedtuple


class Particle(NamedTuple):
    name: str
    x: float
    y: float
    z: float

    def __str__(self):
        return '\t'.join(str(prop) for prop in self)


class Frame(NamedTuple):
    n_particles: int
    comment: str
    particles: list[Particle]

    def __str__(self):
        return '\n'.join([str(self.n_particles), self.comment, 
                          *[str(p) for p in self.particles]])

# Particle = namedtuple('Particle', ['name', 'x', 'y', 'z'])
# Frame = namedtuple('Frame', ['n_particles', 'comment', 'particles'])


def read_xyz(file: Iterable[str]) -> list[Frame]:
    file = iter(file)
    trajectory: list[Frame] = []
    while True:
        try:
            header = next(file).strip()
        except StopIteration:
            return trajectory
        try:
            n_particles: int = int(header)
        except ValueError:
            raise ValueError('Expected a integer denoting the number of particles in that frame, '
                             f'received {header!r}.')
        try:
            comment: str = next(file).strip()

            particles: list[Particle] = []
            for _ in range(n_particles):
                name, x, y, z = next(file).strip().split()
                x, y, z = float(x), float(y), float(z)
                particle = Particle(name, x, y, z)
                particles.append(particle)

            frame = Frame(n_particles, comment, particles)
            trajectory.append(frame)
        except StopIteration:
            raise ValueError('File ended early. '
                             'Number of particles does not match or comment line is missing')

#def write_xyz(trajectory: list[Frame]) -> list[str]:
#    res = []
#    for frame in trajectory:
#        res.append(str(frame.n_particles))
#        res.append(frame.comment)
#        for particle in frame.particles:
#            res.append('\t'.join(str(prop) for prop in particle))
#    return res
