import pytest
from structured_data_profiling.profiler import DatasetProfiler


@pytest.mark.parametrize(
    "data_path",
    [
        (
            pytest.lazy_fixture("lending"),
            "./test/resources/datasets/lending/lending.csv",
        ),
        (pytest.lazy_fixture("adult"), "./test/resources/datasets/adult/uci_adult.csv"),
    ],
)
def test_import(data_path):
    dp = DatasetProfiler(data_path[1])

    assert dp.reduced_data_sample.shape[0] > 0
    assert dp.reduced_data_sample.shape[1] > 0


@pytest.mark.parametrize(
    "data_path",
    [
        (
            pytest.lazy_fixture("lending"),
            "./test/resources/datasets/lending/lending.csv",
        ),
        (pytest.lazy_fixture("adult"), "./test/resources/datasets/adult/uci_adult.csv"),
    ],
)
def test_profile(data_path):
    dp = DatasetProfiler(data_path[1])
    dp.profile()
#    dp.warnings()
    assert dp.reduced_data_sample.shape[0] > 0
    assert dp.reduced_data_sample.shape[1] > 0
