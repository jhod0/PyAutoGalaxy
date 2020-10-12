import autofit as af
import autogalaxy as ag
from autogalaxy import exc

import pytest


class TestSetupHyper:
    def test__hyper_searches(self):
        setup = ag.SetupHyper(hyper_galaxies=False)
        assert setup.hyper_galaxies_search == None

        setup = ag.SetupHyper(hyper_galaxies=True)
        assert setup.hyper_galaxies_search.n_live_points == 75
        assert setup.hyper_galaxies_search.evidence_tolerance == pytest.approx(
            0.084, 1.0e-4
        )

        setup = ag.SetupHyper(
            hyper_galaxies=True,
            hyper_galaxies_search=af.DynestyStatic(n_live_points=51),
        )
        assert setup.hyper_galaxies_search.n_live_points == 51
        assert setup.hyper_galaxies_search.evidence_tolerance == pytest.approx(
            0.06, 1.0e-4
        )

        setup = ag.SetupHyper(inversion_search=None)
        assert setup.inversion_search.n_live_points == 50
        assert setup.inversion_search.evidence_tolerance == pytest.approx(0.059, 1.0e-4)

        setup = ag.SetupHyper(inversion_search=af.DynestyStatic(n_live_points=51))
        assert setup.inversion_search.n_live_points == 51

        setup = ag.SetupHyper(hyper_combined_search=af.DynestyStatic(n_live_points=51))
        assert setup.hyper_combined_search.n_live_points == 51

        setup = ag.SetupHyper(hyper_galaxies=True, evidence_tolerance=0.5)
        assert setup.hyper_galaxies_search.evidence_tolerance == pytest.approx(
            0.084, 1.0e-4
        )
        assert setup.inversion_search.evidence_tolerance == 0.5
        assert setup.hyper_combined_search.evidence_tolerance == 0.5

        with pytest.raises(exc.PipelineException):
            ag.SetupHyper(
                hyper_galaxies=True,
                hyper_galaxies_search=af.DynestyStatic(n_live_points=51),
                evidence_tolerance=0.5,
            )

        with pytest.raises(exc.PipelineException):
            ag.SetupHyper(
                inversion_search=af.DynestyStatic(n_live_points=51),
                evidence_tolerance=3.0,
            )

        with pytest.raises(exc.PipelineException):
            ag.SetupHyper(
                hyper_combined_search=af.DynestyStatic(n_live_points=51),
                evidence_tolerance=3.0,
            )

    def test__hyper_galaxies_tag(self):
        setup = ag.SetupHyper(hyper_galaxies=False)
        assert setup.hyper_galaxies_tag == ""

        setup = ag.SetupHyper(hyper_galaxies=True)
        assert setup.hyper_galaxies_tag == "galaxies"

    def test__hyper_image_sky_tag(self):
        setup = ag.SetupHyper(hyper_image_sky=False)
        assert setup.hyper_galaxies_tag == ""

        setup = ag.SetupHyper(hyper_image_sky=True)
        assert setup.hyper_image_sky_tag == "__bg_sky"

    def test__hyper_background_noise_tag(self):
        setup = ag.SetupHyper(hyper_background_noise=False)
        assert setup.hyper_galaxies_tag == ""

        setup = ag.SetupHyper(hyper_background_noise=True)
        assert setup.hyper_background_noise_tag == "__bg_noise"

    def test__tag(self):

        setup = ag.SetupHyper(
            hyper_galaxies=False, hyper_image_sky=False, hyper_background_noise=False
        )

        assert setup.tag == ""

        setup = ag.SetupHyper(
            hyper_galaxies=True, hyper_image_sky=True, hyper_background_noise=True
        )

        assert setup.tag == "hyper[galaxies__bg_sky__bg_noise]"

        setup = ag.SetupHyper(hyper_galaxies=True, hyper_background_noise=True)

        assert setup.tag == "hyper[galaxies__bg_noise]"


class TestSetupLightParametric:
    def test__prior_models_and_tags(self):

        setup = ag.SetupLightParametric(
            bulge_prior_model=af.PriorModel(ag.lp.EllipticalSersic),
            disk_prior_model=af.PriorModel(ag.lp.EllipticalExponential),
        )

        assert setup.bulge_prior_model.cls is ag.lp.EllipticalSersic
        assert setup.disk_prior_model.cls is ag.lp.EllipticalExponential
        assert setup.envelope_prior_model is None

        assert setup.bulge_prior_model_tag == "__bulge_sersic"
        assert setup.disk_prior_model_tag == "__disk_exp"
        assert setup.envelope_prior_model_tag == ""

        setup = ag.SetupLightParametric(
            bulge_prior_model=af.PriorModel(ag.lp.SphericalSersic),
            disk_prior_model=af.PriorModel(ag.lp.SphericalExponential),
            envelope_prior_model=af.PriorModel(ag.lp.EllipticalCoreSersic),
        )

        assert setup.bulge_prior_model_tag == "__bulge_sersic_sph"
        assert setup.disk_prior_model_tag == "__disk_exp_sph"
        assert setup.envelope_prior_model_tag == "__envelope_core_sersic"

    def test__light_centre_tag(self):

        setup = ag.SetupLightParametric(light_centre=None)
        assert setup.light_centre_tag == ""
        setup = ag.SetupLightParametric(light_centre=(2.0, 2.0))
        assert setup.light_centre_tag == "__centre_(2.00,2.00)"
        setup = ag.SetupLightParametric(light_centre=(3.0, 4.0))
        assert setup.light_centre_tag == "__centre_(3.00,4.00)"
        setup = ag.SetupLightParametric(light_centre=(3.027, 4.033))
        assert setup.light_centre_tag == "__centre_(3.03,4.03)"

    def test__align_centre_to_light_centre(self):

        light = af.PriorModel(ag.mp.SphericalIsothermal)

        source = ag.SetupLightParametric(light_centre=(1.0, 2.0))

        light = source.align_centre_to_light_centre(light_prior_model=light)

        assert light.centre == (1.0, 2.0)

    def test__align_bulge_disk_tags(self):
        light = ag.SetupLightParametric(align_bulge_disk_centre=False)
        assert light.align_bulge_disk_centre_tag == ""
        light = ag.SetupLightParametric(align_bulge_disk_centre=True)
        assert light.align_bulge_disk_centre_tag == "_centre"

        light = ag.SetupLightParametric(align_bulge_disk_elliptical_comps=False)
        assert light.align_bulge_disk_elliptical_comps_tag == ""
        light = ag.SetupLightParametric(align_bulge_disk_elliptical_comps=True)
        assert light.align_bulge_disk_elliptical_comps_tag == "_ell"

    def test__tag(self):

        setup = ag.SetupLightParametric(light_centre=None)
        assert (
            setup.tag
            == "light[parametric__bulge_sersic__disk_exp__align_bulge_disk_centre]"
        )

        setup = ag.SetupLightParametric(
            light_centre=(3.027, 4.033), align_bulge_disk_centre=False
        )
        assert (
            setup.tag == "light[parametric__bulge_sersic__disk_exp__centre_(3.03,4.03)]"
        )

        light = ag.SetupLightParametric(
            align_bulge_disk_centre=True, align_bulge_disk_elliptical_comps=True
        )

        assert (
            light.tag
            == "light[parametric__bulge_sersic__disk_exp__align_bulge_disk_centre_ell]"
        )

        light = ag.SetupLightParametric(
            bulge_prior_model=af.PriorModel(ag.lp.SphericalSersic),
            disk_prior_model=af.PriorModel(ag.lp.SphericalExponential),
            envelope_prior_model=af.PriorModel(ag.lp.EllipticalCoreSersic),
            align_bulge_disk_centre=True,
            align_bulge_disk_elliptical_comps=True,
        )

        assert (
            light.tag
            == "light[parametric__bulge_sersic_sph__disk_exp_sph__envelope_core_sersic__align_bulge_disk_centre_ell]"
        )


class TestSetupLightInversion:
    def test__pixelization_prior_model__model_depends_on_inversion_pixels_fixed(self):

        setup = ag.SetupLightInversion(
            pixelization_prior_model=af.PriorModel(ag.pix.Rectangular),
            regularization_prior_model=af.PriorModel(ag.reg.Regularization),
        )

        assert setup.pixelization_prior_model.cls is ag.pix.Rectangular

        setup = ag.SetupLightInversion(
            pixelization_prior_model=af.PriorModel(ag.pix.VoronoiBrightnessImage),
            regularization_prior_model=af.PriorModel(ag.reg.Regularization),
        )

        assert setup.pixelization_prior_model.cls is ag.pix.VoronoiBrightnessImage

        setup = ag.SetupLightInversion(
            pixelization_prior_model=af.PriorModel(ag.pix.VoronoiBrightnessImage),
            inversion_pixels_fixed=100,
            regularization_prior_model=af.PriorModel(ag.reg.Regularization),
        )

        assert isinstance(setup.pixelization_prior_model, af.PriorModel)
        assert setup.pixelization_prior_model.pixels == 100

    def test__pixelization_tag(self):

        setup = ag.SetupLightInversion(
            pixelization_prior_model=af.PriorModel(ag.pix.Rectangular),
            regularization_prior_model=af.PriorModel(ag.reg.Regularization),
        )
        assert setup.pixelization_tag == "__pix_rect"

        setup = ag.SetupLightInversion(
            pixelization_prior_model=af.PriorModel(ag.pix.VoronoiBrightnessImage),
            regularization_prior_model=af.PriorModel(ag.reg.Regularization),
        )
        assert setup.pixelization_tag == "__pix_voro_image"

    def test__regularization_tag(self):

        setup = ag.SetupLightInversion(
            pixelization_prior_model=af.PriorModel(ag.pix.Pixelization),
            regularization_prior_model=af.PriorModel(ag.reg.Constant),
        )
        assert setup.regularization_tag == "__reg_const"

        setup = ag.SetupLightInversion(
            pixelization_prior_model=af.PriorModel(ag.pix.Pixelization),
            regularization_prior_model=af.PriorModel(ag.reg.AdaptiveBrightness),
        )
        assert setup.regularization_tag == "__reg_adapt_bright"

    def test__inversion_pixels_fixed_tag(self):

        setup = ag.SetupLightInversion(
            inversion_pixels_fixed=None,
            pixelization_prior_model=af.PriorModel(ag.pix.Pixelization),
            regularization_prior_model=af.PriorModel(ag.reg.AdaptiveBrightness),
        )
        assert setup.inversion_pixels_fixed_tag == ""

        setup = ag.SetupLightInversion(
            inversion_pixels_fixed=100,
            pixelization_prior_model=af.PriorModel(ag.pix.Pixelization),
            regularization_prior_model=af.PriorModel(ag.reg.AdaptiveBrightness),
        )
        assert setup.inversion_pixels_fixed_tag == ""

        setup = ag.SetupLightInversion(
            inversion_pixels_fixed=100,
            pixelization_prior_model=af.PriorModel(ag.pix.VoronoiBrightnessImage),
            regularization_prior_model=af.PriorModel(ag.reg.AdaptiveBrightness),
        )
        assert setup.inversion_pixels_fixed_tag == "_100"

    def test__tag(self):

        setup = ag.SetupLightInversion(
            pixelization_prior_model=af.PriorModel(ag.pix.Rectangular),
            regularization_prior_model=af.PriorModel(ag.reg.Constant),
            inversion_pixels_fixed=100,
        )

        assert setup.tag == "light[inversion__pix_rect__reg_const]"

        setup = ag.SetupLightInversion(
            pixelization_prior_model=af.PriorModel(ag.pix.VoronoiBrightnessImage),
            regularization_prior_model=af.PriorModel(ag.reg.AdaptiveBrightness),
            inversion_pixels_fixed=None,
        )

        assert setup.tag == "light[inversion__pix_voro_image__reg_adapt_bright]"

        setup = ag.SetupLightInversion(
            pixelization_prior_model=af.PriorModel(ag.pix.VoronoiBrightnessImage),
            regularization_prior_model=af.PriorModel(ag.reg.AdaptiveBrightness),
            inversion_pixels_fixed=100,
        )

        assert setup.tag == "light[inversion__pix_voro_image_100__reg_adapt_bright]"


class TestAbstractSetupMass:
    def test__align_centre_to_mass_centre(self):

        mass = af.PriorModel(ag.mp.SphericalIsothermal)

        source = ag.SetupMassTotal(mass_centre=(1.0, 2.0))

        mass = source.align_centre_to_mass_centre(mass_prior_model=mass)

        assert mass.centre == (1.0, 2.0)

    def test__unfix_mass_centre(self):

        mass = af.PriorModel(ag.mp.SphericalIsothermal)
        mass.centre = (1.0, 2.0)

        setup_mass = ag.SetupMassTotal()

        mass = af.PriorModel(ag.mp.SphericalIsothermal)

        mass = setup_mass.unfix_mass_centre(mass_prior_model=mass)

        # assert mass.centre.centre_0.mean == 5.0
        # assert mass.centre.centre_0.sigma == 0.05
        # assert mass.centre.centre_1.mean == 6.0
        # assert mass.centre.centre_1.sigma == 0.05


class TestSetupMassTotal:
    def test__mass_prior_model_and_tags(self):

        setup = ag.SetupMassTotal(
            mass_prior_model=af.PriorModel(ag.mp.EllipticalPowerLaw)
        )

        assert setup.mass_prior_model.cls is ag.mp.EllipticalPowerLaw
        assert setup.mass_prior_model_tag == "__power_law"

        setup = ag.SetupMassTotal(
            mass_prior_model=af.PriorModel(ag.mp.EllipticalIsothermal)
        )

        assert setup.mass_prior_model.cls is ag.mp.EllipticalIsothermal
        assert setup.mass_prior_model_tag == "__sie"

    def test__mass_centre_tag(self):
        setup = ag.SetupMassTotal(mass_centre=None)
        assert setup.mass_centre_tag == ""
        setup = ag.SetupMassTotal(mass_centre=(2.0, 2.0))
        assert setup.mass_centre_tag == "__centre_(2.00,2.00)"
        setup = ag.SetupMassTotal(mass_centre=(3.0, 4.0))
        assert setup.mass_centre_tag == "__centre_(3.00,4.00)"
        setup = ag.SetupMassTotal(mass_centre=(3.027, 4.033))
        assert setup.mass_centre_tag == "__centre_(3.03,4.03)"

    def test__align_centre_of_mass_to_light(self):

        mass = af.PriorModel(ag.mp.SphericalIsothermal)

        source = ag.SetupMassTotal(align_light_mass_centre=False)

        mass = source.align_centre_of_mass_to_light(
            mass_prior_model=mass, light_centre=(1.0, 2.0)
        )

        assert mass.centre.centre_0.mean == 1.0
        assert mass.centre.centre_0.sigma == 0.1
        assert mass.centre.centre_0.mean == 1.0
        assert mass.centre.centre_0.sigma == 0.1

        source = ag.SetupMassTotal(align_light_mass_centre=True)

        mass = source.align_centre_of_mass_to_light(
            mass_prior_model=mass, light_centre=(1.0, 2.0)
        )

        assert mass.centre == (1.0, 2.0)

    def test__tag(self):

        setup = ag.SetupMassTotal(mass_centre=None)
        assert setup.tag == "mass[total__power_law]"

        setup = ag.SetupMassTotal(mass_centre=(3.027, 4.033))
        assert setup.tag == "mass[total__power_law__centre_(3.03,4.03)]"

        setup = ag.SetupMassTotal(
            mass_centre=(3.027, 4.033),
            mass_prior_model=af.PriorModel(ag.mp.EllipticalIsothermal),
        )
        assert setup.tag == "mass[total__sie__centre_(3.03,4.03)]"


class TestSetupMassLightDark:
    def test__prior_models_and_tags(self):

        setup = ag.SetupMassLightDark(
            bulge_prior_model=af.PriorModel(ag.lmp.EllipticalSersic),
            disk_prior_model=af.PriorModel(ag.lmp.EllipticalExponential),
        )

        assert setup.bulge_prior_model.cls is ag.lmp.EllipticalSersic
        assert setup.disk_prior_model.cls is ag.lmp.EllipticalExponential
        assert setup.envelope_prior_model is None
        assert setup.dark_prior_model.cls is ag.mp.SphericalNFWMCRLudlow

        assert setup.bulge_prior_model_tag == "__bulge_sersic"
        assert setup.disk_prior_model_tag == "__disk_exp"
        assert setup.envelope_prior_model_tag == ""
        assert setup.dark_prior_model_tag == "__dark_nfw_sph_ludlow"

        setup = ag.SetupMassLightDark(
            bulge_prior_model=af.PriorModel(ag.lmp.EllipticalDevVaucouleurs),
            disk_prior_model=af.PriorModel(ag.lmp.SphericalDevVaucouleurs),
            envelope_prior_model=af.PriorModel(ag.lmp.SphericalExponential),
            dark_prior_model=af.PriorModel(ag.mp.EllipticalNFW),
        )

        assert setup.bulge_prior_model.cls is ag.lmp.EllipticalDevVaucouleurs
        assert setup.disk_prior_model.cls is ag.lmp.SphericalDevVaucouleurs
        assert setup.envelope_prior_model.cls is ag.lmp.SphericalExponential
        assert setup.dark_prior_model.cls is ag.mp.EllipticalNFW

        assert setup.bulge_prior_model_tag == "__bulge_dev"
        assert setup.disk_prior_model_tag == "__disk_dev_sph"
        assert setup.envelope_prior_model_tag == "__envelope_exp_sph"
        assert setup.dark_prior_model_tag == "__dark_nfw"

    def test__constant_mass_to_light_ratio_tag(self):

        setup = ag.SetupMassLightDark(constant_mass_to_light_ratio=True)
        assert setup.constant_mass_to_light_ratio_tag == "__mlr_const"

        setup = ag.SetupMassLightDark(constant_mass_to_light_ratio=False)
        assert setup.constant_mass_to_light_ratio_tag == "__mlr_free"

    def test__align_bulge_dark_tag(self):

        setup = ag.SetupMassLightDark(align_bulge_dark_centre=False)
        assert setup.align_bulge_dark_centre_tag == ""

        setup = ag.SetupMassLightDark(align_bulge_dark_centre=True)
        assert setup.align_bulge_dark_centre_tag == "__align_bulge_dark_centre"

    def test__set_mass_to_light_ratios_of_light_and_mass_profiles(self):

        lmp_0 = af.PriorModel(ag.lmp.EllipticalSersic)
        lmp_1 = af.PriorModel(ag.lmp.EllipticalSersic)
        lmp_2 = af.PriorModel(ag.lmp.EllipticalSersic)

        setup = ag.SetupMassLightDark(constant_mass_to_light_ratio=False)

        setup.set_mass_to_light_ratios_of_light_and_mass_prior_models(
            light_and_mass_prior_models=[lmp_0, lmp_1, lmp_2]
        )

        assert lmp_0.mass_to_light_ratio != lmp_1.mass_to_light_ratio
        assert lmp_0.mass_to_light_ratio != lmp_2.mass_to_light_ratio
        assert lmp_1.mass_to_light_ratio != lmp_2.mass_to_light_ratio

        lmp_0 = af.PriorModel(ag.lmp.EllipticalSersic)
        lmp_1 = af.PriorModel(ag.lmp.EllipticalSersic)
        lmp_2 = af.PriorModel(ag.lmp.EllipticalSersic)

        setup = ag.SetupMassLightDark(constant_mass_to_light_ratio=True)

        setup.set_mass_to_light_ratios_of_light_and_mass_prior_models(
            light_and_mass_prior_models=[lmp_0, lmp_1, lmp_2]
        )

        assert lmp_0.mass_to_light_ratio == lmp_1.mass_to_light_ratio
        assert lmp_0.mass_to_light_ratio == lmp_2.mass_to_light_ratio
        assert lmp_1.mass_to_light_ratio == lmp_2.mass_to_light_ratio

    def test__tag(self):

        setup = ag.SetupMassLightDark(
            bulge_prior_model=af.PriorModel(ag.lmp.EllipticalSersic),
            disk_prior_model=af.PriorModel(ag.lmp.EllipticalExponential),
            constant_mass_to_light_ratio=True,
            align_bulge_dark_centre=True,
        )
        assert (
            setup.tag
            == "mass[light_dark__bulge_sersic__disk_exp__mlr_const__dark_nfw_sph_ludlow__align_bulge_dark_centre]"
        )


class TestSMBH:
    def test__tag(self):

        setup = ag.SetupSMBH(smbh_centre_fixed=True)

        assert setup.tag == "smbh[point_mass__centre_fixed]"

        setup = ag.SetupSMBH(smbh_centre_fixed=False)

        assert setup.tag == "smbh[point_mass__centre_free]"

    def test__smbh_from_centre(self):

        setup = ag.SetupSMBH(smbh_centre_fixed=True)
        smbh = setup.smbh_from_centre(centre=(0.0, 0.0))

        assert isinstance(smbh, af.PriorModel)
        assert smbh.centre == (0.0, 0.0)

        setup = ag.SetupSMBH(smbh_centre_fixed=False)
        smbh = setup.smbh_from_centre(centre=(0.1, 0.2), centre_sigma=0.2)

        assert isinstance(smbh, af.PriorModel)

        assert isinstance(smbh.centre[0], af.GaussianPrior)
        assert smbh.centre[0].mean == 0.1
        assert smbh.centre[0].sigma == 0.2

        assert isinstance(smbh.centre[1], af.GaussianPrior)
        assert smbh.centre[1].mean == 0.2
        assert smbh.centre[1].sigma == 0.2


class TestSetupPipeline:
    def test__setup__passes_light_setup_to_mass_light_setup(self):

        setup_light = ag.SetupLightParametric(
            bulge_prior_model=af.PriorModel(ag.lp.EllipticalSersic),
            disk_prior_model=af.PriorModel(ag.lp.EllipticalExponential),
        )

        setup_mass = ag.SetupMassLightDark()

        setup_pipeline = ag.SetupPipeline(
            setup_light=setup_light, setup_mass=setup_mass
        )

        assert (
            setup_pipeline.setup_light.bulge_prior_model.cls is ag.lp.EllipticalSersic
        )
        assert (
            setup_pipeline.setup_light.disk_prior_model.cls
            is ag.lp.EllipticalExponential
        )
        assert setup_pipeline.setup_light.envelope_prior_model is None

        assert setup_pipeline.setup_mass.bulge_prior_model.cls is ag.lp.EllipticalSersic
        assert (
            setup_pipeline.setup_mass.disk_prior_model.cls
            is ag.lp.EllipticalExponential
        )
        assert setup_pipeline.setup_mass.envelope_prior_model is None

    def test__tag(self):

        setup_light = ag.SetupLightParametric(
            light_centre=(1.0, 2.0), align_bulge_disk_centre=False
        )

        setup = ag.SetupPipeline(setup_light=setup_light)

        assert (
            setup.tag == "setup__"
            "light[parametric__bulge_sersic__disk_exp__centre_(1.00,2.00)]"
        )

        setup_hyper = ag.SetupHyper(hyper_galaxies=True, hyper_background_noise=True)

        setup_light = ag.SetupLightParametric(light_centre=(1.0, 2.0))

        setup = ag.SetupPipeline(setup_hyper=setup_hyper, setup_light=setup_light)

        assert (
            setup.tag == "setup__hyper[galaxies__bg_noise]__"
            "light[parametric__bulge_sersic__disk_exp__align_bulge_disk_centre__centre_(1.00,2.00)]"
        )

        smbh = ag.SetupSMBH(smbh_centre_fixed=True)

        setup = ag.SetupPipeline(setup_smbh=smbh)

        assert setup.tag == "setup__smbh[point_mass__centre_fixed]"
