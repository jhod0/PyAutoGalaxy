import autogalaxy as ag

from test_autogalaxy.aggregator.conftest import clean, aggregator_from

database_file = "db_imaging"


def test__dataset_generator_from_aggregator__analysis_has_single_dataset(
    image_7x7, psf_3x3, noise_map_7x7, mask_2d_7x7, samples, model
):
    imaging = ag.Imaging(
        data=image_7x7,
        psf=psf_3x3,
        noise_map=noise_map_7x7,
        over_sampling=ag.OverSamplingDataset(
            uniform=ag.OverSamplingIterate(fractional_accuracy=0.5, sub_steps=[2]),
            pixelization=ag.OverSamplingUniform(sub_size=3),
        ),
    )

    masked_imaging_7x7 = imaging.apply_mask(mask=mask_2d_7x7)

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)

    agg = aggregator_from(
        database_file=database_file,
        analysis=analysis,
        model=model,
        samples=samples,
    )

    dataset_agg = ag.agg.ImagingAgg(aggregator=agg)
    dataset_gen = dataset_agg.dataset_gen_from()

    for dataset_list in dataset_gen:
        assert (dataset_list[0].data == masked_imaging_7x7.data).all()
        assert isinstance(
            dataset_list[0].grids.uniform.over_sampling, ag.OverSamplingIterate
        )
        assert isinstance(
            dataset_list[0].grids.pixelization.over_sampling, ag.OverSamplingUniform
        )
        assert dataset_list[0].grids.uniform.over_sampling.sub_steps == [2]
        assert dataset_list[0].grids.uniform.over_sampling.fractional_accuracy == 0.5

    clean(database_file=database_file)


def test__dataset_generator_from_aggregator__analysis_multi(
    analysis_imaging_7x7, samples, model
):
    agg = aggregator_from(
        database_file=database_file,
        analysis=analysis_imaging_7x7 + analysis_imaging_7x7,
        model=model,
        samples=samples,
    )

    dataset_agg = ag.agg.ImagingAgg(aggregator=agg)
    dataset_gen = dataset_agg.dataset_gen_from()

    for dataset_list in dataset_gen:
        assert (dataset_list[0].data == analysis_imaging_7x7.dataset.data).all()
        assert (dataset_list[1].data == analysis_imaging_7x7.dataset.data).all()

    clean(database_file=database_file)
