from autoarray.dataset import imaging
from autogalaxy.plane import plane as pl

import numpy as np
import copy


class MaskedImaging(imaging.MaskedImaging):
    def __init__(
        self,
        imaging,
        mask,
        psf_shape_2d=None,
        inversion_pixel_limit=None,
        renormalize_psf=True,
    ):
        """
        The lens dataset is the collection of data_type (image, noise map, PSF), a mask, grid, convolver \
        and other utilities that are used for modeling and fitting an image of a strong lens.

        Whilst the image, noise map, etc. are loaded in 2D, the lens dataset creates reduced 1D arrays of each \
        for lens calculations.

        Parameters
        ----------
        imaging: im.Imaging
            The imaging data_type all in 2D (the image, noise map, PSF, etc.)
        mask: msk.Mask
            The 2D mask that is applied to the image.
        sub_size : int
            The size of the sub-grid used for each lens SubGrid. E.g. a value of 2 grid each image-pixel on a 2x2 \
            sub-grid.
        psf_shape_2d : (int, int)
            The shape of the PSF used for convolving model image generated using analytic light profiles. A smaller \
            shape will trim the PSF relative to the input image PSF, giving a faster analysis run-time.
        positions : [[]]
            Lists of image-pixel coordinates (arc-seconds) that mappers close to one another in the source-plane(s), \
            used to speed up the non-linear sampling.
        pixel_scale_interpolation_grid : float
            If *True*, expensive to compute mass profile deflection angles will be computed on a sparse grid and \
            interpolated to the grid, sub and blurring grids.
        inversion_pixel_limit : int or None
            The maximum number of pixels that can be used by an inversion, with the limit placed primarily to speed \
            up run.
        """

        super(MaskedImaging, self).__init__(
            imaging=imaging,
            mask=mask,
            psf_shape_2d=psf_shape_2d,
            inversion_pixel_limit=inversion_pixel_limit,
            renormalize_psf=renormalize_psf,
        )

    def modify_image_and_noise_map(self, image, noise_map):

        masked_imaging = copy.deepcopy(self)

        masked_imaging.image = image
        masked_imaging.noise_map = noise_map

        return masked_imaging


class SimulatorImaging(imaging.SimulatorImaging):
    def __init__(
        self,
        psf,
        exposure_time_map,
        background_sky_map=None,
        renormalize_psf=True,
        add_noise=True,
        noise_if_add_noise_false=0.1,
        noise_seed=-1,
    ):
        """A class representing a Imaging observation, using the shape of the image, the pixel scale,
        psf, exposure time, etc.

        Parameters
        ----------
        shape_2d : (int, int)
            The shape of the observation. Note that we do not simulator a full Imaging frame (e.g. 2000 x 2000 pixels for \
            Hubble imaging), but instead just a cut-out around the strong lens.
        pixel_scales : float
            The size of each pixel in arc seconds.
        psf : PSF
            An arrays describing the PSF kernel of the image.
        exposure_time_map : float
            The exposure time of an observation using this data_type.
        background_sky_map : float
            The level of the background sky of an observationg using this data_type.
        """

        super(SimulatorImaging, self).__init__(
            psf=psf,
            exposure_time_map=exposure_time_map,
            background_sky_map=background_sky_map,
            renormalize_psf=renormalize_psf,
            add_noise=add_noise,
            noise_if_add_noise_false=noise_if_add_noise_false,
            noise_seed=noise_seed,
        )

    def from_plane_and_grid(self, plane, grid, name=None):
        """
        Create a realistic simulated image by applying effects to a plain simulated image.

        Parameters
        ----------
        name
        image : ndarray
            The image before simulating (e.g. the lens and source galaxies before optics blurring and Imaging read-out).
        pixel_scales: float
            The scale of each pixel in arc seconds
        exposure_time_map : ndarray
            An arrays representing the effective exposure time of each pixel.
        psf: PSF
            An arrays describing the PSF the simulated image is blurred with.
        background_sky_map : ndarray
            The value of background sky in every image pixel (electrons per second).
        add_noise: Bool
            If True poisson noise_maps is simulated and added to the image, based on the total counts in each image
            pixel
        noise_seed: int
            A seed for random noise_maps generation
        """

        image = plane.padded_profile_image_from_grid_and_psf_shape(
            grid=grid, psf_shape_2d=self.psf.shape_2d
        )

        if self.psf is not None:

            simulator = copy.deepcopy(self)
            simulator.exposure_time_map = self.exposure_time_map.padded_from_kernel_shape(
                kernel_shape_2d=self.psf.shape_2d
            )

            if self.background_sky_map is not None:

                simulator.background_sky_map = self.background_sky_map.padded_from_kernel_shape(
                    kernel_shape_2d=self.psf.shape_2d
                )

        else:

            simulator = self

        return simulator.from_image(image=image.in_1d_binned, name=name)

    def from_galaxies_and_grid(self, galaxies, grid, name=None):
        """Simulate imaging data for this data_type, as follows:

        1)  Setup the image-plane grid of the Imaging arrays, which defines the coordinates used for the ray-tracing.

        2) Use this grid and the lens and source galaxies to setup a tracer, which generates the image of \
           the simulated imaging data.

        3) Simulate the imaging data, using a special image which ensures edge-effects don't
           degrade simulator of the telescope optics (e.g. the PSF convolution).

        4) Plot the image using Matplotlib, if the plot_imaging bool is True.

        5) Output the dataset to .fits format if a dataset_path and data_name are specified. Otherwise, return the simulated \
           imaging data_type instance."""

        plane = pl.Plane(redshift=float(np.mean([galaxy.redshift for galaxy in galaxies])), galaxies=galaxies)

        return self.from_plane_and_grid(plane=plane, grid=grid, name=name)
