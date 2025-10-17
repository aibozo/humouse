from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data import loaders, registry  # noqa: E402


def test_list_dataset_ids_contains_expected():
    dataset_ids = registry.list_dataset_ids()
    assert {"balabit", "bogazici", "sapimouse", "attentive_cursor"}.issubset(dataset_ids)


def test_ensure_dataset_path_exists():
    root = registry.get_project_root()
    for dataset_id in registry.list_dataset_ids():
        path = registry.get_dataset_spec(dataset_id).absolute_path(root)
        assert path.exists(), f"Dataset path missing: {dataset_id}"


def test_validate_dataset_returns_results():
    results = loaders.validate_all_datasets()
    assert len(results) == len(registry.list_dataset_ids())
    ids = {result.dataset_id for result in results}
    assert ids == set(registry.list_dataset_ids())


@pytest.mark.parametrize("dataset_id", registry.list_dataset_ids())
def test_validate_dataset_sample_file(dataset_id: str):
    result = loaders.validate_dataset(dataset_id)
    assert result.exists
    if result.sample_file is not None:
        assert result.sample_file.exists()
        spec = registry.get_dataset_spec(dataset_id)
        if spec.expected_columns is not None and result.columns_match is not None:
            assert result.columns_match is True, result.message

