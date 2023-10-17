from typing import Tuple

import autoarray as aa
from autogalaxy.profiles.mass.abstract.abstract import MassProfile

class dPIESph(MassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        ra: float = 0.1,
        rs: float = 2.0,
        sigma_scale: float = 0.1,
    ):
        super(MassProfile, self).__init__(centre, ell_comps)

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid):
        raise NotImplementedError

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def convergence_2d_from(self, grid):
        raise NotImplementedError

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def potential_2d_from(self, grid):
        raise NotImplementedError
