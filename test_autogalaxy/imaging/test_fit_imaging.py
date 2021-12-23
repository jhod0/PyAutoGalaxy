import numpy as np
import pytest

import autogalaxy as ag
from autogalaxy.mock.mock import MockLightProfile


class MockFitImaging:
    def __init__(self, model_images_of_galaxies):

        self.model_images_of_galaxies = model_images_of_galaxies


def test__model_image__with_and_without_psf_blurring(
    masked_imaging_7x7_no_blur, masked_imaging_7x7
):

    g0 = ag.Galaxy(
        redshift=0.5,
        light_profile=MockLightProfile(image_2d_value=1.0, image_2d_first_value=2.0),
    )
    plane = ag.Plane(galaxies=[g0])

    fit = ag.FitImaging(dataset=masked_imaging_7x7_no_blur, plane=plane)

    assert fit.model_image.slim == pytest.approx(
        np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 1.0e-4
    )
    assert fit.log_likelihood == pytest.approx(-14.63377, 1.0e-4)

    fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

    assert fit.model_image.slim == pytest.approx(
        np.array([1.33, 1.16, 1.0, 1.16, 1.0, 1.0, 1.0, 1.0, 1.0]), 1.0e-1
    )
    assert fit.log_likelihood == pytest.approx(-14.52960, 1.0e-4)


def test__noise_map__with_and_without_hyper_galaxy(masked_imaging_7x7_no_blur):

    g0 = ag.Galaxy(
        redshift=0.5,
        light_profile=MockLightProfile(image_2d_value=1.0, image_2d_first_value=2.0),
    )

    plane = ag.Plane(galaxies=[g0])

    fit = ag.FitImaging(dataset=masked_imaging_7x7_no_blur, plane=plane)

    assert fit.noise_map.slim == pytest.approx(
        np.full(fill_value=2.0, shape=(9,)), 1.0e-1
    )

    assert fit.log_likelihood == pytest.approx(-14.6337, 1.0e-4)

    g0 = ag.Galaxy(
        redshift=0.5,
        light_profile=MockLightProfile(image_2d_value=1.0, image_2d_first_value=2.0),
        hyper_galaxy=ag.HyperGalaxy(
            contribution_factor=1.0, noise_factor=1.0, noise_power=1.0
        ),
        hyper_model_image=ag.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0),
        hyper_galaxy_image=ag.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0),
        hyper_minimum_value=0.0,
    )

    plane = ag.Plane(galaxies=[g0])

    fit = ag.FitImaging(dataset=masked_imaging_7x7_no_blur, plane=plane)

    assert fit.noise_map.slim == pytest.approx(
        np.full(fill_value=4.0, shape=(9,)), 1.0e-1
    )

    assert fit.log_likelihood == pytest.approx(-20.7783, 1.0e-4)


def test__hyper_image_changes_background_sky__reflected_in_likelihood():
    psf = ag.Kernel2D.manual_native(
        array=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], pixel_scales=1.0
    )

    imaging = ag.Imaging(
        image=ag.Array2D.full(fill_value=4.0, shape_native=(3, 4), pixel_scales=1.0),
        psf=psf,
        noise_map=ag.Array2D.ones(shape_native=(3, 4), pixel_scales=1.0),
    )
    imaging.image[5] = 5.0

    mask = ag.Mask2D.manual(
        mask=[
            [True, True, True, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=1.0,
    )

    masked_imaging_7x7 = imaging.apply_mask(mask=mask)
    masked_imaging_7x7 = masked_imaging_7x7.apply_settings(
        settings=ag.SettingsImaging(grid_class=ag.Grid2D, sub_size=1)
    )

    # Setup as a ray trace instance, using a light profile for the galaxy

    g0 = ag.Galaxy(redshift=0.5, light_profile=MockLightProfile(image_2d=np.ones(2)))
    plane = ag.Plane(galaxies=[g0])

    hyper_image_sky = ag.hyper_data.HyperImageSky(sky_scale=1.0)

    fit = ag.FitImaging(
        dataset=masked_imaging_7x7, plane=plane, hyper_image_sky=hyper_image_sky
    )

    assert (
        fit.mask
        == np.array(
            [
                [True, True, True, True],
                [True, False, False, True],
                [True, True, True, True],
            ]
        )
    ).all()

    assert (
        fit.image.native
        == np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 6.0, 5.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    ).all()

    assert (
        fit.chi_squared_map.native
        == np.array(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 25.0, 16.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        )
    ).all()

    assert fit.chi_squared == 41.0
    assert fit.reduced_chi_squared == 41.0 / 2.0
    assert fit.noise_normalization == pytest.approx(
        2.0 * np.log(2 * np.pi * 1.0 ** 2.0), 1.0e-4
    )
    assert fit.log_likelihood == pytest.approx(
        -0.5 * (41.0 + 2.0 * np.log(2 * np.pi * 1.0 ** 2.0)), 1.0e-4
    )


def test__hyper_background_changes_background_noise_map__reflected_in_likelihood():
    psf = ag.Kernel2D.manual_native(
        array=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], pixel_scales=1.0
    )

    imaging = ag.Imaging(
        image=5.0 * ag.Array2D.ones(shape_native=(3, 4), pixel_scales=1.0),
        psf=psf,
        noise_map=ag.Array2D.ones(shape_native=(3, 4), pixel_scales=1.0),
    )
    imaging.image[6] = 4.0

    mask = ag.Mask2D.manual(
        mask=[
            [True, True, True, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=1.0,
    )

    masked_imaging_7x7 = imaging.apply_mask(mask=mask)
    masked_imaging_7x7 = masked_imaging_7x7.apply_settings(
        settings=ag.SettingsImaging(grid_class=ag.Grid2D, sub_size=1)
    )

    # Setup as a ray trace instance, using a light profile for the galaxy

    g0 = ag.Galaxy(redshift=0.5, light_profile=MockLightProfile(image_2d=np.ones(2)))
    plane = ag.Plane(galaxies=[g0])

    hyper_background_noise = ag.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

    fit = ag.FitImaging(
        dataset=masked_imaging_7x7,
        plane=plane,
        hyper_background_noise=hyper_background_noise,
    )

    assert (
        fit.noise_map.native
        == np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 2.0, 2.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    ).all()

    assert fit.chi_squared == 6.25
    assert fit.reduced_chi_squared == 6.25 / 2.0
    assert fit.noise_normalization == pytest.approx(
        2.0 * np.log(2 * np.pi * 2.0 ** 2.0), 1.0e-4
    )
    assert fit.log_likelihood == pytest.approx(
        -0.5 * (6.25 + 2.0 * np.log(2 * np.pi * 2.0 ** 2.0)), 1.0e-4
    )


def test__hyper_galaxy_changes_noise_above_hyper_noise_limit__rounded_down_to_limit():
    # This PSF changes the blurred image plane image from [1.0, 1.0] to [1.0, 5.0]

    # Thus, the chi squared is 4.0**2.0 + 0.0**2.0 = 16.0

    # The hyper_galaxies galaxy increases the noise in both pixels by 1.0, to 2.0.

    # This reduces the chi squared to 2.0 instead of 4.0

    psf = ag.Kernel2D.manual_native(
        array=[[0.0, 0.0, 0.0], [0.0, 1.0, 3.0], [0.0, 0.0, 0.0]], pixel_scales=1.0
    )

    imaging = ag.Imaging(
        image=5.0 * ag.Array2D.ones(shape_native=(3, 4), pixel_scales=1.0),
        psf=psf,
        noise_map=ag.Array2D.ones(shape_native=(3, 4), pixel_scales=1.0),
    )
    imaging.image[6] = 4.0

    mask = ag.Mask2D.manual(
        mask=[
            [True, True, True, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=1.0,
    )

    masked_imaging_7x7 = imaging.apply_mask(mask=mask)
    masked_imaging_7x7 = masked_imaging_7x7.apply_settings(
        settings=ag.SettingsImaging(
            grid_class=ag.Grid2D, use_normalized_psf=False, sub_size=1
        )
    )

    # Setup as a ray trace instance, using a light profile for the galaxy

    g0 = ag.Galaxy(
        redshift=0.5,
        light_profile=MockLightProfile(image_2d=np.ones(2)),
        hyper_galaxy=ag.HyperGalaxy(
            contribution_factor=1.0, noise_factor=1.0e9, noise_power=1.0
        ),
        hyper_model_image=ag.Array2D.ones(shape_native=(1, 2), pixel_scales=1.0),
        hyper_galaxy_image=ag.Array2D.ones(shape_native=(1, 2), pixel_scales=1.0),
        hyper_minimum_value=0.0,
    )

    plane = ag.Plane(galaxies=[g0])

    fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

    assert (
        fit.noise_map.native
        == np.array(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0e8, 1.0e8, 0.0], [0.0, 0.0, 0.0, 0.0]]
        )
    ).all()


class TestCompareToManualProfilesOnly:
    def test___all_fit_quantities__no_hyper_methods(self, masked_imaging_7x7):
        g0 = ag.Galaxy(
            redshift=0.5,
            light_profile=ag.lp.EllSersic(intensity=1.0),
            mass_profile=ag.mp.SphIsothermal(einstein_radius=1.0),
        )

        g1 = ag.Galaxy(redshift=1.0, light_profile=ag.lp.EllSersic(intensity=1.0))

        plane = ag.Plane(redshift=0.75, galaxies=[g0, g1])

        fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

        assert masked_imaging_7x7.noise_map.native == pytest.approx(
            fit.noise_map.native
        )

        model_image = plane.blurred_image_2d_via_convolver_from(
            grid=masked_imaging_7x7.grid,
            convolver=masked_imaging_7x7.convolver,
            blurring_grid=masked_imaging_7x7.blurring_grid,
        )

        assert model_image.native == pytest.approx(fit.model_image.native)

        residual_map = ag.util.fit.residual_map_from(
            data=masked_imaging_7x7.image, model_data=model_image
        )

        assert residual_map.native == pytest.approx(fit.residual_map.native)

        normalized_residual_map = ag.util.fit.normalized_residual_map_from(
            residual_map=residual_map, noise_map=masked_imaging_7x7.noise_map
        )

        assert normalized_residual_map.native == pytest.approx(
            fit.normalized_residual_map.native
        )

        chi_squared_map = ag.util.fit.chi_squared_map_from(
            residual_map=residual_map, noise_map=masked_imaging_7x7.noise_map
        )

        assert chi_squared_map.native == pytest.approx(fit.chi_squared_map.native)

        chi_squared = ag.util.fit.chi_squared_from(chi_squared_map=chi_squared_map)

        noise_normalization = ag.util.fit.noise_normalization_from(
            noise_map=masked_imaging_7x7.noise_map
        )

        log_likelihood = ag.util.fit.log_likelihood_from(
            chi_squared=chi_squared, noise_normalization=noise_normalization
        )

        assert log_likelihood == pytest.approx(fit.log_likelihood, 1e-4)
        assert log_likelihood == fit.figure_of_merit

    def test___fit_galaxy_model_image_dict__corresponds_to_blurred_galaxy_images(
        self, masked_imaging_7x7
    ):
        g0 = ag.Galaxy(
            redshift=0.5,
            light_profile=ag.lp.EllSersic(intensity=1.0),
            mass_profile=ag.mp.SphIsothermal(einstein_radius=1.0),
        )
        g1 = ag.Galaxy(redshift=1.0, light_profile=ag.lp.EllSersic(intensity=1.0))
        g2 = ag.Galaxy(redshift=1.0)

        plane = ag.Plane(redshift=0.75, galaxies=[g0, g1, g2])

        fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

        g0_blurred_image_2d = g0.blurred_image_2d_via_convolver_from(
            grid=masked_imaging_7x7.grid,
            blurring_grid=masked_imaging_7x7.blurring_grid,
            convolver=masked_imaging_7x7.convolver,
        )

        g1_blurred_image_2d = g1.blurred_image_2d_via_convolver_from(
            grid=masked_imaging_7x7.grid,
            blurring_grid=masked_imaging_7x7.blurring_grid,
            convolver=masked_imaging_7x7.convolver,
        )

        assert fit.galaxy_model_image_dict[g0] == pytest.approx(
            g0_blurred_image_2d, 1.0e-4
        )
        assert fit.galaxy_model_image_dict[g1] == pytest.approx(
            g1_blurred_image_2d, 1.0e-4
        )
        assert (fit.galaxy_model_image_dict[g2].slim == np.zeros(9)).all()

        assert fit.model_image.native == pytest.approx(
            fit.galaxy_model_image_dict[g0].native
            + fit.galaxy_model_image_dict[g1].native,
            1.0e-4,
        )

    def test___all_fit_quantities__including_hyper_methods(self, masked_imaging_7x7):
        hyper_image_sky = ag.hyper_data.HyperImageSky(sky_scale=1.0)

        hyper_background_noise = ag.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

        image = hyper_image_sky.hyper_image_from(image=masked_imaging_7x7.image)

        g0 = ag.Galaxy(
            redshift=0.5,
            light_profile=ag.lp.EllSersic(intensity=1.0),
            mass_profile=ag.mp.SphIsothermal(einstein_radius=1.0),
            hyper_galaxy=ag.HyperGalaxy(
                contribution_factor=1.0, noise_factor=1.0, noise_power=1.0
            ),
            hyper_model_image=np.ones(9),
            hyper_galaxy_image=np.ones(9),
            hyper_minimum_value=0.0,
        )
        g1 = ag.Galaxy(redshift=1.0, light_profile=ag.lp.EllSersic(intensity=1.0))

        plane = ag.Plane(redshift=0.75, galaxies=[g0, g1])

        fit = ag.FitImaging(
            dataset=masked_imaging_7x7,
            plane=plane,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

        hyper_noise_map_background = hyper_background_noise.hyper_noise_map_from(
            noise_map=masked_imaging_7x7.noise_map
        )

        hyper_noise = plane.hyper_noise_map_from(noise_map=masked_imaging_7x7.noise_map)

        hyper_noise_map = hyper_noise_map_background + hyper_noise

        assert hyper_noise_map.native == pytest.approx(fit.noise_map.native)

        model_image = plane.blurred_image_2d_via_convolver_from(
            grid=masked_imaging_7x7.grid,
            convolver=masked_imaging_7x7.convolver,
            blurring_grid=masked_imaging_7x7.blurring_grid,
        )

        assert model_image.native == pytest.approx(fit.model_image.native)

        residual_map = ag.util.fit.residual_map_from(data=image, model_data=model_image)

        assert residual_map.native == pytest.approx(fit.residual_map.native)

        normalized_residual_map = ag.util.fit.normalized_residual_map_from(
            residual_map=residual_map, noise_map=hyper_noise_map
        )

        assert normalized_residual_map.native == pytest.approx(
            fit.normalized_residual_map.native
        )

        chi_squared_map = ag.util.fit.chi_squared_map_from(
            residual_map=residual_map, noise_map=hyper_noise_map
        )

        assert chi_squared_map.native == pytest.approx(fit.chi_squared_map.native)

        chi_squared = ag.util.fit.chi_squared_from(chi_squared_map=chi_squared_map)

        noise_normalization = ag.util.fit.noise_normalization_from(
            noise_map=hyper_noise_map
        )

        log_likelihood = ag.util.fit.log_likelihood_from(
            chi_squared=chi_squared, noise_normalization=noise_normalization
        )

        assert log_likelihood == pytest.approx(fit.log_likelihood, 1e-4)
        assert log_likelihood == fit.figure_of_merit

        fit = ag.FitImaging(
            dataset=masked_imaging_7x7,
            plane=plane,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
            use_hyper_scalings=False,
        )

        assert fit.image == pytest.approx(masked_imaging_7x7.image, 1.0e-4)
        assert fit.noise_map == pytest.approx(masked_imaging_7x7.noise_map, 1.0e-4)

    def test___blurred_and_model_images_of_galaxies_and_unmasked_blurred_image_properties(
        self, masked_imaging_7x7
    ):
        g0 = ag.Galaxy(
            redshift=0.5,
            light_profile=ag.lp.EllSersic(intensity=1.0),
            mass_profile=ag.mp.SphIsothermal(einstein_radius=1.0),
        )

        g1 = ag.Galaxy(redshift=1.0, light_profile=ag.lp.EllSersic(intensity=1.0))

        plane = ag.Plane(redshift=0.75, galaxies=[g0, g1])

        fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

        g0_blurred_image = g0.blurred_image_2d_via_convolver_from(
            grid=masked_imaging_7x7.grid,
            convolver=masked_imaging_7x7.convolver,
            blurring_grid=masked_imaging_7x7.blurring_grid,
        )
        g1_blurred_image = g1.blurred_image_2d_via_convolver_from(
            grid=masked_imaging_7x7.grid,
            convolver=masked_imaging_7x7.convolver,
            blurring_grid=masked_imaging_7x7.blurring_grid,
        )

        assert g0_blurred_image.native == pytest.approx(
            fit.model_images_of_galaxies[0].native, 1.0e-4
        )

        assert g1_blurred_image.native == pytest.approx(
            fit.model_images_of_galaxies[1].native, 1.0e-4
        )

        unmasked_blurred_image = plane.unmasked_blurred_image_2d_via_psf_from(
            grid=masked_imaging_7x7.grid, psf=masked_imaging_7x7.psf
        )

        assert (unmasked_blurred_image == fit.unmasked_blurred_image).all()

        unmasked_blurred_image_of_galaxies = plane.unmasked_blurred_image_2d_list_via_psf_from(
            grid=masked_imaging_7x7.grid, psf=masked_imaging_7x7.psf
        )

        assert (
            unmasked_blurred_image_of_galaxies[0]
            == fit.unmasked_blurred_image_of_galaxies[0]
        ).all()
        assert (
            unmasked_blurred_image_of_galaxies[1]
            == fit.unmasked_blurred_image_of_galaxies[1]
        ).all()


class TestCompareToManualInversionOnly:
    def test___all_quantities__no_hyper_methods(self, masked_imaging_7x7):
        # Ensures the inversion grid is used, as this would cause the test to fail.
        masked_imaging_7x7.grid[0, 0] = -100.0

        pix = ag.pix.Rectangular(shape=(3, 3))
        reg = ag.reg.Constant(coefficient=1.0)

        g0 = ag.Galaxy(redshift=0.5, pixelization=pix, regularization=reg)

        plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5), g0])

        fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

        mapper = pix.mapper_from(
            source_grid_slim=masked_imaging_7x7.grid_inversion,
            source_pixelization_grid=None,
        )

        inversion = ag.Inversion(
            dataset=masked_imaging_7x7,
            linear_obj_list=[mapper],
            regularization_list=[reg],
        )

        assert inversion.mapped_reconstructed_image.native == pytest.approx(
            fit.model_image.native, 1.0e-4
        )

        residual_map = ag.util.fit.residual_map_from(
            data=masked_imaging_7x7.image,
            model_data=inversion.mapped_reconstructed_image,
        )

        assert residual_map.native == pytest.approx(fit.residual_map.native, 1.0e-4)

        normalized_residual_map = ag.util.fit.normalized_residual_map_from(
            residual_map=residual_map, noise_map=masked_imaging_7x7.noise_map
        )

        assert normalized_residual_map.native == pytest.approx(
            fit.normalized_residual_map.native, 1.0e-4
        )

        chi_squared_map = ag.util.fit.chi_squared_map_from(
            residual_map=residual_map, noise_map=masked_imaging_7x7.noise_map
        )

        assert chi_squared_map.native == pytest.approx(
            fit.chi_squared_map.native, 1.0e-4
        )

        chi_squared = ag.util.fit.chi_squared_from(chi_squared_map=chi_squared_map)

        noise_normalization = ag.util.fit.noise_normalization_from(
            noise_map=masked_imaging_7x7.noise_map
        )

        log_likelihood = ag.util.fit.log_likelihood_from(
            chi_squared=chi_squared, noise_normalization=noise_normalization
        )

        assert log_likelihood == pytest.approx(fit.log_likelihood, 1e-4)

        log_likelihood_with_regularization = ag.util.fit.log_likelihood_with_regularization_from(
            chi_squared=chi_squared,
            regularization_term=inversion.regularization_term,
            noise_normalization=noise_normalization,
        )

        assert log_likelihood_with_regularization == pytest.approx(
            fit.log_likelihood_with_regularization, 1e-4
        )

        log_evidence = ag.util.fit.log_evidence_from(
            chi_squared=chi_squared,
            regularization_term=inversion.regularization_term,
            log_curvature_regularization_term=inversion.log_det_curvature_reg_matrix_term,
            log_regularization_term=inversion.log_det_regularization_matrix_term,
            noise_normalization=noise_normalization,
        )

        assert log_evidence == fit.log_evidence
        assert log_evidence == fit.figure_of_merit

    def test___fit_galaxy_model_image_dict__has_inversion_mapped_reconstructed_image(
        self, masked_imaging_7x7
    ):
        pix = ag.pix.Rectangular(shape=(3, 3))
        reg = ag.reg.Constant(coefficient=1.0)

        g0 = ag.Galaxy(redshift=0.5)
        g1 = ag.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

        plane = ag.Plane(redshift=0.75, galaxies=[g0, g1])

        fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

        mapper = pix.mapper_from(
            source_grid_slim=masked_imaging_7x7.grid, source_pixelization_grid=None
        )

        inversion = ag.Inversion(
            dataset=masked_imaging_7x7,
            linear_obj_list=[mapper],
            regularization_list=[reg],
        )

        assert (fit.galaxy_model_image_dict[g0] == np.zeros(9)).all()

        assert fit.galaxy_model_image_dict[g1].native == pytest.approx(
            inversion.mapped_reconstructed_image.native, 1.0e-4
        )

        assert fit.model_image.native == pytest.approx(
            fit.galaxy_model_image_dict[g1].native, 1.0e-4
        )

    def test___all_fit_quantities__include_hyper_methods(self, masked_imaging_7x7):
        hyper_image_sky = ag.hyper_data.HyperImageSky(sky_scale=1.0)

        hyper_background_noise = ag.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

        image = hyper_image_sky.hyper_image_from(image=masked_imaging_7x7.image)

        hyper_noise_map_background = hyper_background_noise.hyper_noise_map_from(
            noise_map=masked_imaging_7x7.noise_map
        )

        pix = ag.pix.Rectangular(shape=(3, 3))
        reg = ag.reg.Constant(coefficient=1.0)

        g0 = ag.Galaxy(
            redshift=0.5,
            pixelization=pix,
            regularization=reg,
            hyper_galaxy=ag.HyperGalaxy(
                contribution_factor=1.0, noise_factor=1.0, noise_power=1.0
            ),
            hyper_model_image=np.ones(9),
            hyper_galaxy_image=np.ones(9),
            hyper_minimum_value=0.0,
        )

        plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5), g0])

        fit = ag.FitImaging(
            dataset=masked_imaging_7x7,
            plane=plane,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
            settings_inversion=ag.SettingsInversion(use_w_tilde=False),
        )

        hyper_noise = plane.hyper_noise_map_from(noise_map=masked_imaging_7x7.noise_map)
        hyper_noise_map = hyper_noise_map_background + hyper_noise

        assert hyper_noise_map.native == pytest.approx(fit.noise_map.native)

        mapper = pix.mapper_from(
            source_grid_slim=masked_imaging_7x7.grid,
            settings=ag.SettingsPixelization(use_border=False),
        )
        inversion = ag.InversionImaging(
            image=image,
            noise_map=hyper_noise_map,
            convolver=masked_imaging_7x7.convolver,
            w_tilde=masked_imaging_7x7.w_tilde,
            linear_obj_list=[mapper],
            regularization_list=[reg],
            settings=ag.SettingsInversion(use_w_tilde=False),
        )

        assert inversion.mapped_reconstructed_image.native == pytest.approx(
            fit.model_image.native, 1.0e-4
        )

        residual_map = ag.util.fit.residual_map_from(
            data=image, model_data=inversion.mapped_reconstructed_image
        )

        assert residual_map.native == pytest.approx(fit.residual_map.native)

        normalized_residual_map = ag.util.fit.normalized_residual_map_from(
            residual_map=residual_map, noise_map=hyper_noise_map
        )

        assert normalized_residual_map.native == pytest.approx(
            fit.normalized_residual_map.native
        )

        chi_squared_map = ag.util.fit.chi_squared_map_from(
            residual_map=residual_map, noise_map=hyper_noise_map
        )

        assert chi_squared_map.native == pytest.approx(fit.chi_squared_map.native)

        chi_squared = ag.util.fit.chi_squared_from(chi_squared_map=chi_squared_map)

        noise_normalization = ag.util.fit.noise_normalization_from(
            noise_map=hyper_noise_map
        )

        log_likelihood = ag.util.fit.log_likelihood_from(
            chi_squared=chi_squared, noise_normalization=noise_normalization
        )

        assert log_likelihood == pytest.approx(fit.log_likelihood, 1e-4)

        log_likelihood_with_regularization = ag.util.fit.log_likelihood_with_regularization_from(
            chi_squared=chi_squared,
            regularization_term=inversion.regularization_term,
            noise_normalization=noise_normalization,
        )

        assert log_likelihood_with_regularization == pytest.approx(
            fit.log_likelihood_with_regularization, 1e-4
        )

        log_evidence = ag.util.fit.log_evidence_from(
            chi_squared=chi_squared,
            regularization_term=inversion.regularization_term,
            log_curvature_regularization_term=inversion.log_det_curvature_reg_matrix_term,
            log_regularization_term=inversion.log_det_regularization_matrix_term,
            noise_normalization=noise_normalization,
        )

        assert log_evidence == fit.log_evidence
        assert log_evidence == fit.figure_of_merit

    def test___blurred_and_model_images_of_galaxies_and_unmasked_blurred_image_properties(
        self, masked_imaging_7x7
    ):
        pix = ag.pix.Rectangular(shape=(3, 3))
        reg = ag.reg.Constant(coefficient=1.0)

        g0 = ag.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

        plane = ag.Plane(redshift=0.75, galaxies=[ag.Galaxy(redshift=0.5), g0])

        fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

        mapper = pix.mapper_from(
            source_grid_slim=masked_imaging_7x7.grid,
            settings=ag.SettingsPixelization(use_border=False),
        )

        inversion = ag.Inversion(
            dataset=masked_imaging_7x7,
            linear_obj_list=[mapper],
            regularization_list=[reg],
        )

        assert (fit.model_images_of_galaxies[0].native == np.zeros((7, 7))).all()
        assert inversion.mapped_reconstructed_image.native == pytest.approx(
            fit.model_images_of_galaxies[1].native, 1.0e-4
        )


class TestCompareToManualProfilesAndInversion:
    def test___all_fit_quantities__no_hyper_methods(self, masked_imaging_7x7):
        galaxy_light = ag.Galaxy(
            redshift=0.5, light_profile=ag.lp.EllSersic(intensity=1.0)
        )

        pix = ag.pix.Rectangular(shape=(3, 3))
        reg = ag.reg.Constant(coefficient=1.0)
        galaxy_pix = ag.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

        plane = ag.Plane(redshift=0.75, galaxies=[galaxy_light, galaxy_pix])

        fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

        blurred_image = plane.blurred_image_2d_via_convolver_from(
            grid=masked_imaging_7x7.grid,
            convolver=masked_imaging_7x7.convolver,
            blurring_grid=masked_imaging_7x7.blurring_grid,
        )

        assert blurred_image.native == pytest.approx(fit.blurred_image.native)

        profile_subtracted_image = masked_imaging_7x7.image - blurred_image

        assert profile_subtracted_image.native == pytest.approx(
            fit.profile_subtracted_image.native
        )

        mapper = pix.mapper_from(
            source_grid_slim=masked_imaging_7x7.grid,
            settings=ag.SettingsPixelization(use_border=False),
        )

        inversion = ag.InversionImaging(
            image=profile_subtracted_image,
            noise_map=masked_imaging_7x7.noise_map,
            convolver=masked_imaging_7x7.convolver,
            w_tilde=masked_imaging_7x7.w_tilde,
            linear_obj_list=[mapper],
            regularization_list=[reg],
        )

        model_image = blurred_image + inversion.mapped_reconstructed_image

        assert model_image.native == pytest.approx(fit.model_image.native)

        residual_map = ag.util.fit.residual_map_from(
            data=masked_imaging_7x7.image, model_data=model_image
        )

        assert residual_map.native == pytest.approx(fit.residual_map.native)

        normalized_residual_map = ag.util.fit.normalized_residual_map_from(
            residual_map=residual_map, noise_map=masked_imaging_7x7.noise_map
        )

        assert normalized_residual_map.native == pytest.approx(
            fit.normalized_residual_map.native
        )

        chi_squared_map = ag.util.fit.chi_squared_map_from(
            residual_map=residual_map, noise_map=masked_imaging_7x7.noise_map
        )

        assert chi_squared_map.native == pytest.approx(fit.chi_squared_map.native)

        chi_squared = ag.util.fit.chi_squared_from(chi_squared_map=chi_squared_map)

        noise_normalization = ag.util.fit.noise_normalization_from(
            noise_map=masked_imaging_7x7.noise_map
        )

        log_likelihood = ag.util.fit.log_likelihood_from(
            chi_squared=chi_squared, noise_normalization=noise_normalization
        )

        assert log_likelihood == pytest.approx(fit.log_likelihood, 1e-4)

        log_likelihood_with_regularization = ag.util.fit.log_likelihood_with_regularization_from(
            chi_squared=chi_squared,
            regularization_term=inversion.regularization_term,
            noise_normalization=noise_normalization,
        )

        assert log_likelihood_with_regularization == pytest.approx(
            fit.log_likelihood_with_regularization, 1e-4
        )

        log_evidence = ag.util.fit.log_evidence_from(
            chi_squared=chi_squared,
            regularization_term=inversion.regularization_term,
            log_curvature_regularization_term=inversion.log_det_curvature_reg_matrix_term,
            log_regularization_term=inversion.log_det_regularization_matrix_term,
            noise_normalization=noise_normalization,
        )

        assert log_evidence == fit.log_evidence
        assert log_evidence == fit.figure_of_merit

    def test___fit_galaxy_model_image_dict__has_blurred_images_and_inversion_mapped_reconstructed_image(
        self, masked_imaging_7x7
    ):
        g0 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.EllSersic(intensity=1.0))
        g1 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.EllSersic(intensity=2.0))
        g2 = ag.Galaxy(redshift=0.5)

        pix = ag.pix.Rectangular(shape=(3, 3))
        reg = ag.reg.Constant(coefficient=1.0)
        galaxy_pix = ag.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

        plane = ag.Plane(redshift=0.75, galaxies=[g0, g1, g2, galaxy_pix])

        masked_imaging_7x7.image[0] = 3.0

        fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

        g0_blurred_image = g0.blurred_image_2d_via_convolver_from(
            grid=masked_imaging_7x7.grid,
            convolver=masked_imaging_7x7.convolver,
            blurring_grid=masked_imaging_7x7.blurring_grid,
        )
        g1_blurred_image = g1.blurred_image_2d_via_convolver_from(
            grid=masked_imaging_7x7.grid,
            convolver=masked_imaging_7x7.convolver,
            blurring_grid=masked_imaging_7x7.blurring_grid,
        )

        blurred_image = g0_blurred_image + g1_blurred_image

        profile_subtracted_image = masked_imaging_7x7.image - blurred_image
        mapper = pix.mapper_from(
            source_grid_slim=masked_imaging_7x7.grid,
            settings=ag.SettingsPixelization(use_border=False),
        )

        inversion = ag.InversionImaging(
            image=profile_subtracted_image,
            noise_map=masked_imaging_7x7.noise_map,
            convolver=masked_imaging_7x7.convolver,
            w_tilde=masked_imaging_7x7.w_tilde,
            linear_obj_list=[mapper],
            regularization_list=[reg],
        )

        assert (fit.galaxy_model_image_dict[g2] == np.zeros(9)).all()

        assert fit.galaxy_model_image_dict[g0].native == pytest.approx(
            g0_blurred_image.native, 1.0e-4
        )
        assert fit.galaxy_model_image_dict[g1].native == pytest.approx(
            g1_blurred_image.native, 1.0e-4
        )
        assert fit.galaxy_model_image_dict[galaxy_pix].native == pytest.approx(
            inversion.mapped_reconstructed_image.native, 1.0e-4
        )

        assert fit.model_image.native == pytest.approx(
            fit.galaxy_model_image_dict[g0].native
            + fit.galaxy_model_image_dict[g1].native
            + inversion.mapped_reconstructed_image.native,
            1.0e-4,
        )

    def test___all_fit_quantities__include_hyper_methods(self, masked_imaging_7x7):
        hyper_image_sky = ag.hyper_data.HyperImageSky(sky_scale=1.0)

        hyper_background_noise = ag.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

        image = hyper_image_sky.hyper_image_from(image=masked_imaging_7x7.image)

        hyper_noise_map_background = hyper_background_noise.hyper_noise_map_from(
            noise_map=masked_imaging_7x7.noise_map
        )

        galaxy_light = ag.Galaxy(
            redshift=0.5,
            light_profile=ag.lp.EllSersic(intensity=1.0),
            hyper_galaxy=ag.HyperGalaxy(
                contribution_factor=1.0, noise_factor=1.0, noise_power=1.0
            ),
            hyper_model_image=ag.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0),
            hyper_galaxy_image=ag.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0),
            hyper_minimum_value=0.0,
        )

        pix = ag.pix.Rectangular(shape=(3, 3))
        reg = ag.reg.Constant(coefficient=1.0)
        galaxy_pix = ag.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

        plane = ag.Plane(redshift=0.75, galaxies=[galaxy_light, galaxy_pix])

        fit = ag.FitImaging(
            dataset=masked_imaging_7x7,
            plane=plane,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
            settings_inversion=ag.SettingsInversion(use_w_tilde=False),
        )

        hyper_noise = plane.hyper_noise_map_from(noise_map=masked_imaging_7x7.noise_map)
        hyper_noise_map = hyper_noise_map_background + hyper_noise

        assert hyper_noise_map.native == pytest.approx(fit.noise_map.native, 1.0e-4)

        blurred_image = plane.blurred_image_2d_via_convolver_from(
            grid=masked_imaging_7x7.grid,
            convolver=masked_imaging_7x7.convolver,
            blurring_grid=masked_imaging_7x7.blurring_grid,
        )

        assert blurred_image.native == pytest.approx(fit.blurred_image.native)

        profile_subtracted_image = image - blurred_image

        assert profile_subtracted_image.native == pytest.approx(
            fit.profile_subtracted_image.native
        )

        mapper = pix.mapper_from(
            source_grid_slim=masked_imaging_7x7.grid,
            settings=ag.SettingsPixelization(use_border=False),
        )

        inversion = ag.InversionImaging(
            image=profile_subtracted_image,
            noise_map=hyper_noise_map,
            convolver=masked_imaging_7x7.convolver,
            w_tilde=masked_imaging_7x7.w_tilde,
            linear_obj_list=[mapper],
            regularization_list=[reg],
            settings=ag.SettingsInversion(use_w_tilde=False),
        )

        model_image = blurred_image + inversion.mapped_reconstructed_image

        assert model_image.native == pytest.approx(fit.model_image.native, 1.0e-4)

        residual_map = ag.util.fit.residual_map_from(data=image, model_data=model_image)

        assert residual_map.native == pytest.approx(fit.residual_map.native, 1.0e-4)

        normalized_residual_map = ag.util.fit.normalized_residual_map_from(
            residual_map=residual_map, noise_map=hyper_noise_map
        )

        assert normalized_residual_map.native == pytest.approx(
            fit.normalized_residual_map.native, 1.0e-4
        )

        chi_squared_map = ag.util.fit.chi_squared_map_from(
            residual_map=residual_map, noise_map=hyper_noise_map
        )

        assert chi_squared_map.native == pytest.approx(
            fit.chi_squared_map.native, 1.0e-4
        )

        chi_squared = ag.util.fit.chi_squared_from(chi_squared_map=chi_squared_map)

        noise_normalization = ag.util.fit.noise_normalization_from(
            noise_map=hyper_noise_map
        )

        log_likelihood = ag.util.fit.log_likelihood_from(
            chi_squared=chi_squared, noise_normalization=noise_normalization
        )

        assert log_likelihood == pytest.approx(fit.log_likelihood, 1e-4)

        log_likelihood_with_regularization = ag.util.fit.log_likelihood_with_regularization_from(
            chi_squared=chi_squared,
            regularization_term=inversion.regularization_term,
            noise_normalization=noise_normalization,
        )

        assert log_likelihood_with_regularization == pytest.approx(
            fit.log_likelihood_with_regularization, 1e-4
        )

        log_evidence = ag.util.fit.log_evidence_from(
            chi_squared=chi_squared,
            regularization_term=inversion.regularization_term,
            log_curvature_regularization_term=inversion.log_det_curvature_reg_matrix_term,
            log_regularization_term=inversion.log_det_regularization_matrix_term,
            noise_normalization=noise_normalization,
        )

        assert log_evidence == fit.log_evidence
        assert log_evidence == fit.figure_of_merit

    def test___blurred_and_model_images_of_galaxies_and_unmasked_blurred_image_properties(
        self, masked_imaging_7x7
    ):
        galaxy_light = ag.Galaxy(
            redshift=0.5, light_profile=ag.lp.EllSersic(intensity=1.0)
        )

        pix = ag.pix.Rectangular(shape=(3, 3))
        reg = ag.reg.Constant(coefficient=1.0)
        galaxy_pix = ag.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

        plane = ag.Plane(redshift=0.75, galaxies=[galaxy_light, galaxy_pix])

        fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

        blurred_image = plane.blurred_image_2d_via_convolver_from(
            grid=masked_imaging_7x7.grid,
            convolver=masked_imaging_7x7.convolver,
            blurring_grid=masked_imaging_7x7.blurring_grid,
        )

        profile_subtracted_image = masked_imaging_7x7.image - blurred_image

        mapper = pix.mapper_from(
            source_grid_slim=masked_imaging_7x7.grid,
            settings=ag.SettingsPixelization(use_border=False),
        )

        inversion = ag.InversionImaging(
            image=profile_subtracted_image,
            noise_map=masked_imaging_7x7.noise_map,
            convolver=masked_imaging_7x7.convolver,
            w_tilde=masked_imaging_7x7.w_tilde,
            linear_obj_list=[mapper],
            regularization_list=[reg],
        )

        assert blurred_image.native == pytest.approx(
            fit.model_images_of_galaxies[0].native, 1.0e-4
        )
        assert inversion.mapped_reconstructed_image.native == pytest.approx(
            fit.model_images_of_galaxies[1].native, 1.0e-4
        )


class TestAttributes:
    def test__subtracted_images_of_galaxies(self, masked_imaging_7x7_no_blur):

        g0 = ag.Galaxy(
            redshift=0.5, light_profile=MockLightProfile(image_2d=np.ones(1))
        )

        g1 = ag.Galaxy(
            redshift=1.0, light_profile=MockLightProfile(image_2d=2.0 * np.ones(1))
        )

        g2 = ag.Galaxy(
            redshift=1.0, light_profile=MockLightProfile(image_2d=3.0 * np.ones(1))
        )

        plane = ag.Plane(redshift=0.75, galaxies=[g0, g1, g2])

        fit = ag.FitImaging(dataset=masked_imaging_7x7_no_blur, plane=plane)

        assert fit.subtracted_images_of_galaxies[0].slim[0] == -4.0
        assert fit.subtracted_images_of_galaxies[1].slim[0] == -3.0
        assert fit.subtracted_images_of_galaxies[2].slim[0] == -2.0

        g0 = ag.Galaxy(
            redshift=0.5, light_profile=MockLightProfile(image_2d=np.ones(1))
        )

        g1 = ag.Galaxy(redshift=0.5)

        g2 = ag.Galaxy(
            redshift=1.0, light_profile=MockLightProfile(image_2d=3.0 * np.ones(1))
        )

        plane = ag.Plane(redshift=0.75, galaxies=[g0, g1, g2])

        fit = ag.FitImaging(dataset=masked_imaging_7x7_no_blur, plane=plane)

        assert fit.subtracted_images_of_galaxies[0].slim[0] == -2.0
        assert fit.subtracted_images_of_galaxies[1].slim[0] == -3.0
        assert fit.subtracted_images_of_galaxies[2].slim[0] == 0.0
