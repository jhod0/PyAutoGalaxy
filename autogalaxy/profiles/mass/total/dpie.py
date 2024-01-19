from typing import Tuple
import numpy as np

import autoarray as aa
from autogalaxy.profiles.mass.abstract.abstract import MassProfile


class dPIESph(MassProfile):
    '''
    The dual Pseudo-Isothermal Elliptical mass distribution introduced in
    Eliasdottir 2007: https://arxiv.org/abs/0710.5636

    This version is without ellipticity, so perhaps the "E" is a misnomer.

    Corresponds to a projected density profile that looks like:

        \\Sigma(R) = \\Sigma_0 (ra * rs) / (rs - ra) *
                      (1 / \\sqrt(ra^2 + R^2) - 1 / \\sqrt(rs^2 + R^2))

    (c.f. Eliasdottir '07 eqn. A3)

    In this parameterization, ra and rs are the scale radii above in angular
    units (arcsec). The parameter `kappa_scale` is \\Sigma_0 / \\Sigma_crit.
    '''
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ra: float = 0.1,
        rs: float = 2.0,
        kappa_scale: float = 0.1,
    ):
        # Ensure rs > ra (things will probably break otherwise)
        if ra > rs:
            ra, rs = rs, ra
        super(MassProfile, self).__init__(centre, (0.0, 0.0))
        self.ra = ra
        self.rs = rs
        self.kappa_scale = kappa_scale

    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        ys, xs = grid.T
        (ycen, xcen) = self.centre
        xoff, yoff = xs - xcen, ys - ycen
        radii = np.sqrt(xoff**2 + yoff**2)

        r_ra = radii / self.ra
        r_rs = radii / self.rs
        # c.f. Eliasdottir '07 eq. A20
        f = (
            r_ra / (1 + np.sqrt(1 + r_ra*r_ra))
            - r_rs / (1 + np.sqrt(1 + r_rs*r_rs))
        )

        ra, rs = self.ra, self.rs
        # c.f. Eliasdottir '07 eq. A19
        # magnitude of deflection
        alpha = 2 * self.kappa_scale * ra * rs/(rs - ra) * f

        # now we decompose the deflection into y/x components
        defl_y = alpha * yoff / radii
        defl_x = alpha * xoff / radii
        return aa.Grid2DIrregular.from_yx_1d(
            defl_y, defl_x
        )

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def convergence_2d_from(self, grid: aa.type.Grid2DLike):
        # already transformed to center on profile centre so this works
        radsq = (grid[:, 0]**2 + grid[:, 1]**2)
        a, s = self.ra, self.rs
        # c.f. Eliasdottir '07 eqn (A3)
        return (
            self.kappa_scale * (a * s) / (s - a)
            * (1/np.sqrt(a**2 + radsq) - 1/np.sqrt(s**2 + radsq))
        )

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.relocate_to_radial_minimum
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        raise NotImplementedError


class dPIE(MassProfile):
    '''
    The dual Pseudo-Isothermal Elliptical mass distribution introduced in
    Eliasdottir 2007: https://arxiv.org/abs/0710.5636

    Corresponds to a projected density profile that looks like:

        \\Sigma(R) = \\Sigma_0 (ra * rs) / (rs - ra) *
                      (1 / \\sqrt(ra^2 + R^2) - 1 / \\sqrt(rs^2 + R^2))

    (c.f. Eliasdottir '07 eqn. A3)

    In this parameterization, ra and rs are the scale radii above in angular
    units (arcsec). The parameter is \\Sigma_0 / \\Sigma_crit.
    '''
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        ra: float = 0.1,
        rs: float = 2.0,
        sigma_scale: float = 0.1,
    ):
        super(MassProfile, self).__init__(centre, ell_comps)
        self.ra = ra
        self.rs = rs
        self.sigma_scale = sigma_scale

    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        raise NotImplementedError

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def convergence_2d_from(self, grid: aa.type.Grid2DLike):
        raise NotImplementedError

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        raise NotImplementedError
