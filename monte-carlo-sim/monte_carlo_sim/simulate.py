import concurrent.futures as cf
import math
from os import PathLike
from pathlib import Path
import random
import tomllib 

import matplotlib.pyplot as plt
from scipy import constants

from monte_carlo_sim.handle_xyz import Frame, Particle, read_xyz


def simulate(start, topology, steps, max_steps=None, T: float = 300.):
    if not max_steps:
        max_steps = 10*steps

    first_frame = Frame(
            n_particles=start.n_particles,
            comment='step 0',
            particles=start.particles,
        )
    trajectory = [first_frame]
    last_energy = get_conformation_energy(first_frame.particles, topology)

    for _ in range(max_steps):
        trial_structure = perturbe_structure(trajectory[-1].particles)
        trial_energy = get_conformation_energy(trial_structure, topology)

        if metropolis_criterion(
                old_energy=last_energy, 
                new_energy=trial_energy, 
                T=T,
            ):
            last_energy = trial_energy
            n_frame = len(trajectory)
            new_frame = Frame(
                    n_particles=first_frame.n_particles,
                    comment=f'step {n_frame}',
                    particles=trial_structure,
                )
            trajectory.append(new_frame)
            if n_frame >= steps:
                break
    return trajectory


def perturbe_structure(particles: list[Particle], step_length: float = 0.1) -> list[Particle]:
    particles = particles.copy()
    
    x, y, z = random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)
    length = math.sqrt(x*x + y*y + z*z) / step_length

    idx = random.randrange(len(particles))
    chosen = particles[idx]
    perturbed = Particle(
            name=chosen.name,
            x=chosen.x+x/length,
            y=chosen.y+y/length,
            z=chosen.z+z/length,
        )
    particles[idx] = perturbed
    return particles


def metropolis_criterion(new_energy, old_energy, T: float = 300.):
    de = new_energy - old_energy
    if de <= 0:
        return True 
    metropolis = math.exp(-de / (constants.k * T))
    return metropolis > random.random()


def calculate_distance_squared(points: list[Particle]) -> list[list[float]]:
    dist_squared = [[0. for _ in points] for _ in points]
    for i, a in enumerate(points[:-1]):
       for j, b in enumerate(points[i:], start=i):
           r2 = (b.x - a.x)**2 + (b.y - a.y)**2 + (b.z - a.z)**2
           dist_squared[i][j] = dist_squared[j][i] = r2
    return dist_squared


def lennard_jones(r_squared: float, r_min: float, eps: float) -> float:
    r6 = r_min**6 / r_squared**3
    return eps * r6 * (r6 - 2)


def get_conformation_energy(points: list[Particle], parameters: dict[str, list[list[float]]]) -> float:
    distances = calculate_distance_squared(points)
    energy = 0.
    for i, sub in enumerate(distances):
        for j, r2 in enumerate(sub):
            try:
                energy += lennard_jones(
                        r_squared=r2,
                        r_min=parameters['r_min'][i][j],
                        eps=parameters['eps'][i][j],
                    )
            except ZeroDivisionError:
                continue
    return energy


def load_topology(parameter_file: str | PathLike, coordinates: list[Particle]) -> dict[str, list[list[float]]]:
    with open(parameter_file, 'rb') as pf:
        params = tomllib.load(pf)

    t_eps = [[0. for _ in coordinates] for _ in coordinates]
    t_r = t_eps.copy()
    for i, a in enumerate(coordinates):
       for j, b in enumerate(coordinates[i:], start=i):
           t_eps[i][j] = t_eps[j][i] = math.sqrt(
                   params[a.name]['eps'] * params[b.name]['eps']
                )
           t_r[i][j] = t_r[j][i] = (params[a.name]['r_min'] + params[b.name]['r_min']) / 2
    topology = {
            'r_min' : t_r,
            'eps': t_eps,
        }
    return topology


def load_start_structure(filename: str | PathLike, 
                         parameter_file: str | PathLike = Path(__file__).parent / 'parameters.toml'):
    with open(filename, 'r') as cf:
        coordinates = read_xyz(cf)[0]
    topology = load_topology(parameter_file, coordinates.particles)
    return coordinates, topology


def main():
    start, top = load_start_structure('tests/start.xyz')
    with cf.ProcessPoolExecutor() as executor:
        fs = [executor.submit(simulate, start, top, 1_000_000) for _ in range(6)]
        for n, future in enumerate(cf.as_completed(fs)):
            traj = future.result()
            with open(f'out_{n}.xyz', 'w') as traj_file:
                traj_file.write('\n'.join(str(frame) for frame in traj))
            en = [get_conformation_energy(frame.particles, top) for frame in traj]
            plt.plot(en, '.', label=f'run {n}')
    plt.show()


if __name__ == '__main__':
    main()
