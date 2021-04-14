from itertools import count

import numpy as np
from autoarray.inversion import pixelizations as pix, regularization as reg
from autoarray.structures.arrays import values
from autoarray.structures.grids.two_d import grid_2d_irregular
from autoarray.structures.grids import grid_decorators
from autofit.mapper.model_object import ModelObject
from autogalaxy import exc
from autogalaxy import lensing
from autogalaxy.profiles import point_sources as ps
from autogalaxy.profiles import light_profiles as lp
from autogalaxy.profiles import mass_profiles as mp
from autogalaxy.profiles.mass_profiles import (
    dark_mass_profiles as dmp,
    stellar_mass_profiles as smp,
)

from typing import Optional


def is_point_source(obj):
    return isinstance(obj, ps.PointSource)


def is_light_profile(obj):
    return isinstance(obj, lp.LightProfile)


def is_mass_profile(obj):
    return isinstance(obj, mp.MassProfile)


class Galaxy(ModelObject, lensing.LensingObject):
    """
    @DynamicAttrs
    """

    def __init__(
        self,
        redshift: float,
        pixelization: Optional[pix.Pixelization] = None,
        regularization: Optional[reg.Regularization] = None,
        hyper_galaxy: Optional["HyperGalaxy"] = None,
        **kwargs,
    ):
        """Class representing a galaxy, which is composed of attributes used for fitting hyper_galaxies (e.g. light profiles, \
        mass profiles, pixelizations, etc.).
        
        All *has_* methods retun `True` if galaxy has that attribute, `False` if not.

        Parameters
        ----------
        redshift: float
            The redshift of the galaxy.
        light_profiles: [lp.LightProfile]
            A list of the galaxy's light profiles.
        mass_profiles: [mp.MassProfile]
            A list of the galaxy's mass profiles.
        hyper_galaxy : HyperGalaxy
            The hyper_galaxies-parameters of the hyper_galaxies-galaxy, which is used for performing a hyper_galaxies-analysis on the noise-map.
            
        Attributes
        ----------
        pixelization : inversion.Pixelization
            The pixelization of the galaxy used to reconstruct an observed image using an inversion.
        regularization : inversion.Regularization
            The regularization of the pixel-grid used to reconstruct an observed using an inversion.
        """
        super().__init__()
        self.redshift = redshift

        self.hyper_model_image = None
        self.hyper_galaxy_image = None

        for name, val in kwargs.items():
            setattr(self, name, val)

        self.pixelization = pixelization
        self.regularization = regularization

        if pixelization is not None and regularization is None:
            raise exc.GalaxyException(
                "If the galaxy has a pixelization, it must also have a regularization."
            )
        if pixelization is None and regularization is not None:
            raise exc.GalaxyException(
                "If the galaxy has a regularization, it must also have a pixelization."
            )

        self.hyper_galaxy = hyper_galaxy

    def __hash__(self):
        return self.id

    @property
    def point_source_dict(self):
        return {
            key: value for key, value in self.__dict__.items() if is_point_source(value)
        }

    @property
    def light_profiles(self):
        return [value for value in self.__dict__.values() if is_light_profile(value)]

    @property
    def mass_profiles(self):
        return [value for value in self.__dict__.values() if is_mass_profile(value)]

    @property
    def has_redshift(self):
        return self.redshift is not None

    @property
    def has_pixelization(self):
        return self.pixelization is not None

    @property
    def has_regularization(self):
        return self.regularization is not None

    @property
    def has_hyper_galaxy(self):
        return self.hyper_galaxy is not None

    @property
    def has_light_profile(self):
        return len(self.light_profiles) > 0

    @property
    def has_mass_profile(self):
        return len(self.mass_profiles) > 0

    @property
    def has_profile(self):
        return len(self.mass_profiles) + len(self.light_profiles) > 0

    def extract_attribute(self, cls, attr_name):
        """
        Returns an attribute of a class and its children profiles in the the galaxy as a `ValueIrregular`
        or `Grid2DIrregular` object.

        For example, if a galaxy has two light profiles and we want the `LightProfile` axis-ratios, the following:

        `galaxy.extract_attribute(cls=LightProfile, name="axis_ratio"`

        would return:

        ValuesIrregular(values=[axis_ratio_0, axis_ratio_1])

        If a galaxy has three mass profiles and we want the `MassProfile` centres, the following:

        `galaxy.extract_attribute(cls=MassProfile, name="centres"`

         would return:

        GridIrregular2D(grid=[(centre_y_0, centre_x_0), (centre_y_1, centre_x_1), (centre_y_2, centre_x_2)])

        This is used for visualization, for example plotting the centres of all light profiles colored by their profile.
        """

        def extract(value, name):

            try:
                return getattr(value, name)
            except (AttributeError, IndexError):
                return None

        attributes = [
            extract(value, attr_name)
            for value in self.__dict__.values()
            if isinstance(value, cls)
        ]

        attributes = list(filter(None, attributes))

        if attributes == []:
            return None
        elif isinstance(attributes[0], float):
            return values.ValuesIrregular(values=attributes)
        elif isinstance(attributes[0], tuple):
            return grid_2d_irregular.Grid2DIrregular(grid=attributes)

    @property
    def uses_cluster_inversion(self):
        return type(self.pixelization) is pix.VoronoiBrightnessImage

    @property
    def has_stellar_profile(self):
        return len(self.stellar_profiles) > 0

    @property
    def has_dark_profile(self):
        return len(self.dark_profiles) > 0

    @property
    def stellar_profiles(self):
        return [
            profile
            for profile in self.mass_profiles
            if isinstance(profile, smp.StellarProfile)
        ]

    @property
    def dark_profiles(self):
        return [
            profile
            for profile in self.mass_profiles
            if isinstance(profile, dmp.DarkProfile)
        ]

    def stellar_mass_angular_within_circle(self, radius: float):
        if self.has_stellar_profile:
            return sum(
                [
                    profile.mass_angular_within_circle(radius=radius)
                    for profile in self.stellar_profiles
                ]
            )
        else:
            raise exc.GalaxyException(
                "You cannot perform a stellar mass-based calculation on a galaxy which does not have a stellar mass-profile"
            )

    def dark_mass_angular_within_circle(self, radius: float):
        if self.has_dark_profile:
            return sum(
                [
                    profile.mass_angular_within_circle(radius=radius)
                    for profile in self.dark_profiles
                ]
            )
        else:
            raise exc.GalaxyException(
                "You cannot perform a dark mass-based calculation on a galaxy which does not have a dark mass-profile"
            )

    def stellar_fraction_at_radius(self, radius):
        return 1.0 - self.dark_fraction_at_radius(radius=radius)

    def dark_fraction_at_radius(self, radius):

        stellar_mass = self.stellar_mass_angular_within_circle(radius=radius)
        dark_mass = self.dark_mass_angular_within_circle(radius=radius)

        return dark_mass / (stellar_mass + dark_mass)

    def __repr__(self):
        string = "Redshift: {}".format(self.redshift)
        if self.pixelization:
            string += "\nPixelization:\n{}".format(str(self.pixelization))
        if self.regularization:
            string += "\nRegularization:\n{}".format(str(self.regularization))
        if self.hyper_galaxy:
            string += "\nHyper Galaxy:\n{}".format(str(self.hyper_galaxy))
        if self.light_profiles:
            string += "\nLight Profiles:\n{}".format(
                "\n".join(map(str, self.light_profiles))
            )
        if self.mass_profiles:
            string += "\nMass Profiles:\n{}".format(
                "\n".join(map(str, self.mass_profiles))
            )
        return string

    def __eq__(self, other):
        return all(
            (
                isinstance(other, Galaxy),
                self.pixelization == other.pixelization,
                self.redshift == other.redshift,
                self.hyper_galaxy == other.hyper_galaxy,
                self.light_profiles == other.light_profiles,
                self.mass_profiles == other.mass_profiles,
            )
        )

    @grid_decorators.grid_1d_to_structure
    def image_1d_from_grid(self, grid):
        """
        Returns the summed 1D image of all of the galaxy's light profiles using an input grid of Cartesian (y,x)
        coordinates.

        If the galaxy has no light profiles, a grid of zeros is returned.

        See `profiles.light_profiles` for a description of how light profile images are computed.

        Parameters
        ----------
        grid : grid_like
            The (y, x) coordinates in the original reference frame of the grid.

        """
        if self.has_light_profile:
            return sum(
                map(lambda p: p.image_1d_from_grid(grid=grid), self.light_profiles)
            )
        return np.zeros((grid.shape[0],))

    @grid_decorators.grid_2d_to_structure
    def image_2d_from_grid(self, grid):
        """
        Returns the summed 2D image of all of the galaxy's light profiles using an input grid of Cartesian (y,x)
        coordinates.
        
        If the galaxy has no light profiles, a grid of zeros is returned.
        
        See `profiles.light_profiles` for a description of how light profile images are computed.

        Parameters
        ----------
        grid : grid_like
            The (y, x) coordinates in the original reference frame of the grid.

        """
        if self.has_light_profile:
            return sum(
                map(lambda p: p.image_2d_from_grid(grid=grid), self.light_profiles)
            )
        return np.zeros((grid.shape[0],))

    def blurred_image_2d_from_grid_and_psf(self, grid, psf, blurring_grid=None):

        image = self.image_2d_from_grid(grid=grid)

        blurring_image = self.image_2d_from_grid(grid=blurring_grid)

        return psf.convolved_array_from_array_and_mask(
            array=image.binned.native + blurring_image.binned.native, mask=grid.mask
        )

    def blurred_image_2d_from_grid_and_convolver(self, grid, convolver, blurring_grid):

        image = self.image_2d_from_grid(grid=grid)

        blurring_image = self.image_2d_from_grid(grid=blurring_grid)

        return convolver.convolve_image(
            image=image.binned.slim, blurring_image=blurring_image.binned.slim
        )

    def profile_visibilities_from_grid_and_transformer(self, grid, transformer):

        image = self.image_2d_from_grid(grid=grid)

        return transformer.visibilities_from_image(image=image.binned.slim)

    def luminosity_within_circle(self, radius: float):
        """
        Returns the total luminosity of the galaxy's light profiles within a circle of specified radius.

            See *light_profiles.luminosity_within_circle* for details of how this is performed.

            Parameters
            ----------
            radius : float
                The radius of the circle to compute the dimensionless mass within.
            unit_luminosity : str
                The unit_label the luminosity is returned in {esp, counts}.
            exposure_time : float
                The exposure time of the observation, which converts luminosity from electrons per second unit_label to counts.
        """
        if self.has_light_profile:
            return sum(
                map(
                    lambda p: p.luminosity_within_circle(radius=radius),
                    self.light_profiles,
                )
            )

    @grid_decorators.grid_1d_to_structure
    def convergence_1d_from_grid(self, grid):
        """
        Returns the summed 1D convergence of the galaxy's mass profiles using a grid of Cartesian (y,x) coordinates.

        If the galaxy has no mass profiles, a grid of zeros is returned.

        See `profiles.mass_profiles` module for details of how this is performed.

        The `grid_1d_to_structure` decorator reshapes the NumPy arrays the convergence is outputted on. See
        `aa.grid_1d_to_structure` for a description of the output.

        Parameters
        ----------
        grid : grid_like
            The (y, x) coordinates in the original reference frame of the grid.

        """
        if self.has_mass_profile:
            return sum(
                map(lambda p: p.convergence_1d_from_grid(grid=grid), self.mass_profiles)
            )
        return np.zeros((grid.shape[0],))

    @grid_decorators.grid_2d_to_structure
    def convergence_2d_from_grid(self, grid):
        """
        Returns the summed 2D convergence of the galaxy's mass profiles using a grid of Cartesian (y,x) coordinates.

        If the galaxy has no mass profiles, a grid of zeros is returned.
        
        See `profiles.mass_profiles` module for details of how this is performed.

        The `grid_2d_to_structure` decorator reshapes the NumPy arrays the convergence is outputted on. See
        `aa.grid_2d_to_structure` for a description of the output.

        Parameters
        ----------
        grid : grid_like
            The (y, x) coordinates in the original reference frame of the grid.

        """
        if self.has_mass_profile:
            return sum(
                map(lambda p: p.convergence_2d_from_grid(grid=grid), self.mass_profiles)
            )
        return np.zeros((grid.shape[0],))

    @grid_decorators.grid_1d_to_structure
    def potential_1d_from_grid(self, grid):
        """
        Returns the summed 2D gravitational potential of the galaxy's mass profiles using a grid of 
        Cartesian (y,x) coordinates.

        If the galaxy has no mass profiles, a grid of zeros is returned.

        See `profiles.mass_profiles` module for details of how this is performed.

        The `grid_2d_to_structure` decorator reshapes the NumPy arrays the convergence is outputted on. See 
        `aa.grid_2d_to_structure` for a description of the output.

        Parameters
        ----------
        grid : grid_like
            The (y, x) coordinates in the original reference frame of the grid.
        """
        if self.has_mass_profile:
            return sum(
                map(lambda p: p.potential_1d_from_grid(grid=grid), self.mass_profiles)
            )
        return np.zeros((grid.shape[0],))

    @grid_decorators.grid_2d_to_structure
    def potential_2d_from_grid(self, grid):
        """
        Returns the summed 2D gravitational potential of the galaxy's mass profiles using a grid of 
        Cartesian (y,x) coordinates.

        If the galaxy has no mass profiles, a grid of zeros is returned.

        See `profiles.mass_profiles` module for details of how this is performed.

        The `grid_2d_to_structure` decorator reshapes the NumPy arrays the convergence is outputted on. See 
        `aa.grid_2d_to_structure` for a description of the output.

        Parameters
        ----------
        grid : grid_like
            The (y, x) coordinates in the original reference frame of the grid.
        """
        if self.has_mass_profile:
            return sum(
                map(lambda p: p.potential_2d_from_grid(grid=grid), self.mass_profiles)
            )
        return np.zeros((grid.shape[0],))

    @grid_decorators.grid_2d_to_structure
    def deflections_2d_from_grid(self, grid):
        """
        Returns the summed (y,x) deflection angles of the galaxy's mass profiles \
        using a grid of Cartesian (y,x) coordinates.

        If the galaxy has no mass profiles, two grid of zeros are returned.

        See *profiles.mass_profiles* module for details of how this is performed.

        Parameters
        ----------
        grid : grid_like
            The (y, x) coordinates in the original reference frame of the grid.
        """
        if self.has_mass_profile:
            return sum(
                map(lambda p: p.deflections_2d_from_grid(grid=grid), self.mass_profiles)
            )
        return np.zeros((grid.shape[0], 2))

    def mass_angular_within_circle(self, radius: float):
        """ Integrate the mass profiles's convergence profile to compute the total mass within a circle of \
        specified radius. This is centred on the mass profile.

        The following unit_label for mass can be specified and output:

        - Dimensionless angular unit_label (default) - 'angular'.
        - Solar masses - 'angular' (multiplies the angular mass by the critical surface mass density).

        Parameters
        ----------
        radius : dim.Length
            The radius of the circle to compute the dimensionless mass within.
        unit_mass : str
            The unit_label the mass is returned in {angular, angular}.
        critical_surface_density : float or None
            The critical surface mass density of the strong lens configuration, which converts mass from angulalr \
            unit_label to phsical unit_label (e.g. solar masses).
        """
        if self.has_mass_profile:
            return sum(
                map(
                    lambda p: p.mass_angular_within_circle(radius=radius),
                    self.mass_profiles,
                )
            )
        else:
            raise exc.GalaxyException(
                "You cannot perform a mass-based calculation on a galaxy which does not have a mass-profile"
            )

    @property
    def contribution_map(self):
        """
    Returns the contribution map of a galaxy, which represents the fraction of
        flux in each pixel that the galaxy is attributed to contain, hyper to the
        *contribution_factor* hyper_galaxies-parameter.

        This is computed by dividing that galaxy's flux by the total flux in that \
        pixel and then scaling by the maximum flux such that the contribution map \
        ranges between 0 and 1.

        Parameters
        -----------
        hyper_model_image : np.ndarray
            The best-fit model image to the observed image from a previous analysis
            search. This provides the total light attributed to each image pixel by the
            model.
        hyper_galaxy_image : np.ndarray
            A model image of the galaxy (from light profiles or an inversion) from a
            previous analysis search.
        """
        return self.hyper_galaxy.contribution_map_from_hyper_images(
            hyper_model_image=self.hyper_model_image,
            hyper_galaxy_image=self.hyper_galaxy_image,
        )


class HyperGalaxy:
    _ids = count()

    def __init__(self, contribution_factor=0.0, noise_factor=0.0, noise_power=1.0):
        """ If a `Galaxy` is given a *HyperGalaxy* as an attribute, the noise-map in \
        the regions of the image that the galaxy is located will be hyper, to prevent \
        over-fitting of the galaxy.
        
        This is performed by first computing the hyper_galaxies-galaxy's 'contribution-map', \
        which determines the fraction of flux in every pixel of the image that can be \
        associated with this particular hyper_galaxies-galaxy. This is computed using \
        hyper_galaxies-hyper_galaxies set (e.g. fitting.fit_data.FitDataHyper), which includes  best-fit \
        unblurred_image_1d of the galaxy's light from a previous analysis search.
         
        The *HyperGalaxy* class contains the hyper_galaxies-parameters which are associated \
        with this galaxy for scaling the noise-map.
        
        Parameters
        -----------
        contribution_factor : float
            Factor that adjusts how much of the galaxy's light is attributed to the
            contribution map.
        noise_factor : float
            Factor by which the noise-map is increased in the regions of the galaxy's
            contribution map.
        noise_power : float
            The power to which the contribution map is raised when scaling the
            noise-map.
        """
        self.contribution_factor = contribution_factor
        self.noise_factor = noise_factor
        self.noise_power = noise_power

        self.component_number = next(self._ids)

    def contribution_map_from_hyper_images(self, hyper_model_image, hyper_galaxy_image):
        """
        Returns the contribution map of a galaxy, which represents the fraction of
        flux in each pixel that the galaxy is attributed to contain, hyper to the
        *contribution_factor* hyper_galaxies-parameter.

        This is computed by dividing that galaxy's flux by the total flux in that \
        pixel and then scaling by the maximum flux such that the contribution map \
        ranges between 0 and 1.

        Parameters
        -----------
        hyper_model_image : np.ndarray
            The best-fit model image to the observed image from a previous analysis
            search. This provides the total light attributed to each image pixel by the
            model.
        hyper_galaxy_image : np.ndarray
            A model image of the galaxy (from light profiles or an inversion) from a
            previous analysis search.
        """
        contribution_map = np.divide(
            hyper_galaxy_image, np.add(hyper_model_image, self.contribution_factor)
        )
        return np.divide(contribution_map, np.max(contribution_map))

    def hyper_noise_map_from_hyper_images_and_noise_map(
        self, hyper_model_image, hyper_galaxy_image, noise_map
    ):
        contribution_map = self.contribution_map_from_hyper_images(
            hyper_model_image=hyper_model_image, hyper_galaxy_image=hyper_galaxy_image
        )

        return self.hyper_noise_map_from_contribution_map(
            noise_map=noise_map, contribution_map=contribution_map
        )

    def hyper_noise_map_from_contribution_map(self, noise_map, contribution_map):
        """
        Returns a hyper galaxy hyper_galaxies noise-map from a baseline noise-map.

            This uses the galaxy contribution map and the *noise_factor* and *noise_power*
            hyper_galaxies-parameters.

            Parameters
            -----------
            noise_map : np.ndarray
                The observed noise-map (before scaling).
            contribution_map : np.ndarray
                The galaxy contribution map.
        """
        return self.noise_factor * (noise_map * contribution_map) ** self.noise_power

    def __eq__(self, other):
        if isinstance(other, HyperGalaxy):
            return (
                self.contribution_factor == other.contribution_factor
                and self.noise_factor == other.noise_factor
                and self.noise_power == other.noise_power
            )
        return False

    def __str__(self):
        return "\n".join(["{}: {}".format(k, v) for k, v in self.__dict__.items()])


class Redshift(float):
    def __new__(cls, redshift):
        # noinspection PyArgumentList
        return float.__new__(cls, redshift)

    def __init__(self, redshift):
        float.__init__(redshift)
