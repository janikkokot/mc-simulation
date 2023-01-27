from monte_carlo_sim.handle_xyz import read_xyz, Frame, Particle

import pytest


def test_single_frame():
    xyz = """
    3
    This is a comment line
    A 5.67 -3.45 2.61
    B 3.91 -1.91 4
    A 3.2 1.2 -12.3""".strip().splitlines()
    out = list(read_xyz(xyz))
    assert len(out) == 1


def test_output():
    xyz = """
    3
    This is a comment line
    A 5.67 -3.45 2.61
    B 3.91 -1.91 4
    A 3.2 1.2 -12.3""".strip().splitlines()

    out = read_xyz(xyz)
    assert isinstance(out, list)
    assert all(isinstance(frame, Frame) for frame in out)


def test_multiple_frame():
    xyz = """
    3
    Frame 1
    A 5.67 -3.45 2.61
    B 3.91 -1.91 4
    A 3.2 1.2 -12.3
    4
    Frame 2
    B 5.47 -3.45 2.61
    B 3.91 -1.93 3.1
    A 3.2 1.2 -22.4
    A 3.2 1.2 -12.3
    3
    Frame 3
    1 5.67 -3.45 2.61
    1 3.91 -1.91 4
    2 3.2 1.2 -12.3""".strip().splitlines()
    out = read_xyz(xyz)
    assert len(out) == 3


def test_failure_input_length():
    xyz = """
    3
    This is a comment line
    A 5.67 -3.45 2.61
    A 3.2 1.2 -12.3""".strip().splitlines()
    with pytest.raises(ValueError):
        read_xyz(xyz)


def test_failure_input_header():
    xyz = """
    3.2
    This is a comment line
    A 5.67 -3.45 2.61
    B 3.91 -1.91 4
    A 3.2 1.2 -12.3""".strip().splitlines()
    with pytest.raises(TypeError):
        read_xyz(xyz)


def test_frame():
    xyz = """
    3
    This is a comment line
    A 5.67 -3.45 2.61
    B 3.91 -1.91 4
    A 3.2 1.2 -12.3""".strip().splitlines()

    out = read_xyz(xyz)[0]
    assert out.n_particles == len(out.particles)
    assert out.comment == 'This is a comment line'
    assert all(isinstance(p, Particle) for p in out.particles)


def test_particle():
    xyz = """
    3
    This is a comment line
    A 5.67 -3.45 2.61
    B 3.91 -1.91 4
    A 3.2 1.2 -12.3""".strip().splitlines()

    out = read_xyz(xyz)[0]
    particles = out.particles
    assert all(hasattr(p, q) for p in particles for q in 'xyz')
