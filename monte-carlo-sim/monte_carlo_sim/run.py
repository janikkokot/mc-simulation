from argparse import ArgumentParser

from pathlib import Path

from monte_carlo_sim import distance as dist
from monte_carlo_sim import handle_xyz, simulate


def create_parser():
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
                       default=(0.35, 100), type=tuple,
                       help='Parameters for the addaptive step size scaling. \
                             The RATIO denotes the fraction of steps that \
                             should be accepted. The INTERVAL denotes the \
                             frequency at which the step size is updated. \
                             (default: %(default)s)'
                       )

    parser.add_argument('--density', type=float, default=None,
                        metavar='PARTICLES/NM³',
                        help='Density of the simulation box in particles/nm³ \
                              If density is provided, periodic boundary \
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
    parser.add_argument('--append', action='store_true', default=False,
                        help='Append to outputfile instead of overwriting it \
                              (default: %(default)s)',
                        )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    start_particles, topology = handle_xyz.load_start_structure(
            xyz_file=args.c,
            parameter_file=args.parameters,
            )

    distance_fn = (dist.periodic_squared(args.density, len(start_particles))
                   if args.density else dist.non_periodic_squared)

    step_size = (simulate.plain_step_size(args.step_size) if args.fixed else
                 simulate.addaptive_step_size(args.step_size, *args.addaptive))

    trajectory = simulate.monte_carlo(
            start_coordinates=start_particles,
            topology=topology,
            steps=args.steps,
            temperature=args.temperature,
            distance_squared=distance_fn,
            get_step_size=step_size,
            )
    with open(args.x, 'w' if not args.append else 'a') as traj_file:
        traj_file.write('\n'.join(str(frame) for frame in trajectory))


if __name__ == '__main__':
    main()
