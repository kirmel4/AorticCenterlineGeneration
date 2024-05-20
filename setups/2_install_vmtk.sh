PYTHON_MAJOR_VERSION=$(cut -d'.' -f1-2 <<<  $PYTHON_VERSION)
ENV_PYTHON=${ANEURYSM_WORKSPACE}/.venv/lib/python${PYTHON_MAJOR_VERSION}

cd $ANEURYSM_WORKSPACE/setups

docker build --build-arg ANEURYSM_WORKSPACE=${ANEURYSM_WORKSPACE} --build-arg PYTHON_VERSION=${PYTHON_VERSION} --build-arg PYTHON_MAJOR_VERSION=${PYTHON_MAJOR_VERSION} -f vmtk/vmtk.docker -t vmtk .
docker create --name vmtk -i vmtk

docker cp vmtk:/export/site-packages $ENV_PYTHON
docker cp vmtk:/export/vtk.lib/ $ENV_PYTHON/site-packages/
docker cp vmtk:/export/vtkvmtk.lib/ $ENV_PYTHON/site-packages/
docker cp vmtk:/export/itk.lib/ $ENV_PYTHON/site-packages/

docker rm -f vmtk
