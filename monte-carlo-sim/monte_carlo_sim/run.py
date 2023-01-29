from argparse import ArgumentParser
import sys
import itertools
from pathlib import Path

import tomli

from monte_carlo_sim import distance as dist
from monte_carlo_sim import handle_xyz as xyz
from monte_carlo_sim import simulate


def create_parser(args):
    parser = ArgumentParser(
            prog='montecarlo',
            description='Run Monte Carlo Simulations from XYZ-files',
            epilog="Created for 'Introduction to Python'",
            )

    parser.add_argument('-c',
                        required=True, metavar='XYZ',
                        type=Path,
                        help='Start structure for the simulation',
                        )
    parser.add_argument('-x',
                        required=True, metavar='XYZ',
                        type=Path,
                        help='Output file for the trajectory',
                        )
    parser.add_argument('--temperature',
                        required=False, metavar='KELVIN',
                        default=300, type=float,
                        help='Simulation temperature (default: %(default)s)',
                        )
    parser.add_argument('--steps',
                        required=False,
                        default=1_000, type=int,
                        help='Number of steps in the trajectory \
                              (default: %(default)s)',
                        )
    parser.add_argument('--stepsize',
                        metavar='ANGSTROM',
                        default=0.001, type=float,
                        help='Initial step size (default: %(default)s)',
                        )
    steps = parser.add_mutually_exclusive_group()
    steps.add_argument('--fixed', action='store_true',
                       help='If present no addaptive scaling of the step size',
                       )
    steps.add_argument('--addaptive', nargs=2,
                       required=False, metavar=('RATIO', 'INTERVAL'),
                       default=(0.35, 100), type=float,
                       help='Parameters for the addaptive step size scaling. \
                             The RATIO denotes the fraction of steps that \
                             should be accepted. The INTERVAL denotes the \
                             frequency at which the step size is updated. \
                             (default: %(default)s)'
                       )

    parser.add_argument('--stride',
                        metavar='FRAME',
                        default=1, type=int,
                        help="Every n'th frame will be written to the \
                              trajectory %(default)s",
                        )

    periodic = parser.add_mutually_exclusive_group()
    periodic.add_argument('--density', type=float, default=None,
                          metavar='PARTICLES/NM³',
                          help='Density of the simulation box in particles/nm³ \
                                If density is provided, periodic boundary \
                                conditions will be used, otherwise not. \
                                (default: %(default)s)',
                          )
    periodic.add_argument('--box', type=float, default=None,
                          metavar='ANGSTROM',
                          help='Boxsize of simulation box in Angstrom. If \
                                boxsize is provided, periodic boundary \
                                conditions will be used, otherwise not. \
                                (default: %(default)s)',
                          )

    parser.add_argument('--parameters',
                        required=False,
                        metavar='TOML',
                        default=Path(__file__).parent / 'parameters.toml',
                        type=Path,
                        help='Optional parameter file from which the \
                              parameters for the simulation will be taken.',
                        )
    parser.add_argument('--restart', action='store_true', default=False,
                        help='Append to outputfile instead of overwriting it \
                              and use last frame of inputfile as start \
                              structure. (default: %(default)s)',
                        )
    return parser.parse_args(args)


def main():
    args = create_parser(sys.argv[1:])

    # read parameter file
    with open(args.parameters, 'rb') as parameter_file:
        parameters = tomli.load(parameter_file)

    # read coordinate file
    with open(args.c, 'r') as coordinate_file:
        frames = xyz.read_xyz(coordinate_file)
        start_frame = frames[0] if not args.restart else frames[-1]
    start_particles = start_frame.particles

    # generate matching topology
    topology = xyz.generate_topology(
            coordinates=start_particles,
            parameters=parameters,
            )

    # choose proper functions based on command line arguments
    if args.density:
        distance_fn = dist.periodic_squared(args.density, len(start_particles))
    elif args.box:
        density = len(start_particles) / (args.box**3 * 1e-3)
        distance_fn = dist.periodic_squared(density, len(start_particles))
    else:
        distance_fn = dist.non_periodic_squared

    step_size = (simulate.plain_step_size(args.stepsize) if args.fixed else
                 simulate.addaptive_step_size(args.stepsize, *args.addaptive))

    # calculate trajectory using monte carlo
    trajectory = simulate.monte_carlo(
            start_coordinates=start_particles,
            topology=topology,
            steps=args.steps,
            temperature=args.temperature,
            distance_squared=distance_fn,
            get_step_size=step_size,
            )
    with open(args.x, 'w' if not args.restart else 'a') as traj_file:
        for frame in itertools.islice(trajectory, None, None, args.stride):
            traj_file.write(f'{frame}\n')


if __name__ == '__main__':
    main()
