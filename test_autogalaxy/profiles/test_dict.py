import pytest
from autoconf.dictable import as_dict, from_dict

import autogalaxy as ag
from autogalaxy.profiles.geometry_profiles import GeometryProfile


@pytest.fixture(name="ell_sersic")
def make_ell_sersic():
    return ag.mp.Sersic()


@pytest.fixture(name="ell_sersic_dict")
def make_ell_sersic_dict():
    return {
        "type": "instance",
        "class_path": "autogalaxy.profiles.mass.stellar.sersic.Sersic",
        "arguments": {
            "centre": (0.0, 0.0),
            "ell_comps": (0.0, 0.0),
            "intensity": 0.1,
            "effective_radius": 0.6,
            "sersic_index": 0.6,
            "mass_to_light_ratio": 1.0,
        },
    }


def test_to_dict(ell_sersic, ell_sersic_dict):
    assert as_dict(ell_sersic) == ell_sersic_dict


def test_from_dict(ell_sersic, ell_sersic_dict):
    assert ell_sersic == from_dict(ell_sersic_dict)
