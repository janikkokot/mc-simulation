from decimal import DivisionByZero
from os import PathLike
from pathlib import Path
import math
import tomllib 
from typing import NamedTuple

from monte_carlo_sim.handle_xyz import Particle, read_xyz


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
