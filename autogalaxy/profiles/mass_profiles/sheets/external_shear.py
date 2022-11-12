import numpy as np
from scipy.interpolate import griddata
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.mass_profiles.base.abstract import MassProfile

from autogalaxy import convert
from autogalaxy import exc


class ExternalShear(MassProfile):
    def __init__(self, elliptical_comps: Tuple[float, float] = (0.0, 0.0)):
        """
        An `ExternalShear` term, to model the line-of-sight contribution of other galaxies / satellites.

        The shear angle is defined in the direction of stretching of the image. Therefore, if an object located \
        outside the lens is responsible for the shear, it will be offset 90 degrees from the value of angle.

        Parameters
        ----------
        magnitude
            The overall magnitude of the shear (gamma).
        angle
            The rotation axis of the shear.
        """

        super().__init__(centre=(0.0, 0.0), elliptical_comps=elliptical_comps)

    @property
    def magnitude(self):
        return convert.shear_magnitude_from(elliptical_comps=self.elliptical_comps)

    @property
    def angle(self):
        return convert.shear_angle_from(elliptical_comps=self.elliptical_comps)

    def convergence_func(self, grid_radius: float) -> float:
        return 0.0

    def average_convergence_of_1_radius(self):
        return 0.0

    @aa.grid_dec.grid_2d_to_structure
    def convergence_2d_from(self, grid: aa.type.Grid2DLike):
        return np.zeros(shape=grid.shape[0])

    @aa.grid_dec.grid_2d_to_structure
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        return np.zeros(shape=grid.shape[0])

    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """
        deflection_y = -np.multiply(self.magnitude, grid[:, 0])
        deflection_x = np.multiply(self.magnitude, grid[:, 1])
        return self.rotate_grid_from_reference_frame(
            np.vstack((deflection_y, deflection_x)).T
        )
