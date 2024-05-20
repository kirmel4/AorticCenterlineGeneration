mkdir -p /export
cp -r /vmtk-build/Install/lib/python${PYTHON_MAJOR_VERSION}/site-packages/ /export

mkdir -p /export/vtk.lib
cp /vmtk-build/Install/lib/libvtk*.so* /export/vtk.lib

mkdir -p /export/vtkvmtk.lib
cp /vmtk-build/Install/lib/libvtkvmtk*.so /export/vtkvmtk.lib

mkdir -p /export/itk.lib
cp /vmtk-build/Install/lib/libitk*.so* /export/itk.lib
cp /vmtk-build/Install/lib/libITK*.so* /export/itk.lib

ANEURYSM_WORKSPACE=$1
PYTHON_VERSION=$2

ENV_PACKAGES=${ANEURYSM_WORKSPACE}/.venv/lib/python${PYTHON_MAJOR_VERSION}/site-packages
RUNPATH=${ENV_PACKAGES}/vtk.lib:${ENV_PACKAGES}/vtkvmtk.lib:${ENV_PACKAGES}/itk.lib

echo ${ENV_PACKAGES}

for FILE in $(find /export -name '*.so' -or -name '*.so.1'); do
    patchelf --set-rpath ${RUNPATH} ${FILE}
done

