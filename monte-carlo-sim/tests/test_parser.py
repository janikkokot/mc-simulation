from pathlib import Path
import pytest

from monte_carlo_sim.run import create_parser

BASE = ['-c', 'test.xyz', '-x', 'out.xyz']


class TestBase:
    args = create_parser(BASE)

    @pytest.mark.parametrize('var', ['c', 'x', 'parameters'])
    def test_paths(self, var):
        assert isinstance(getattr(self.args, var), Path)

    @pytest.mark.parametrize('var',
                             ['temperature', 'steps', 'stride'],
                             )
    def test_existence_attr(self, var):
        assert getattr(self.args, var)

    @pytest.mark.parametrize('var',
                             ['fixed', 'box', 'density'])
    def test_flag_defaults(self, var):
        assert not getattr(self.args, var)

def test_fixed():
    args = create_parser([*BASE, '--fixed'])
    assert args.fixed
