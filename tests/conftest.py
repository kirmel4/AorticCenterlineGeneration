import pytest

import os

@pytest.fixture
def is_defined_workspace():
    return ('ANEURYSM_WORKSPACE' in os.environ)

@pytest.fixture(autouse=True)
def skip_by_undefined_workspace(request, is_defined_workspace):
    if request.node.get_closest_marker('skip_undefined_workspace'):
        if not is_defined_workspace:
            pytest.skip('No aneurysm workspace defined')
