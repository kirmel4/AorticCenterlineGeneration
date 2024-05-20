import pytest

import os

from pathlib import Path

@pytest.fixture()
def testdata_dirpath():
    workspace_dirpath = Path(os.environ['ANEURYSM_WORKSPACE'])
    dirpath = workspace_dirpath / 'tests/data/geometry'

    return dirpath
