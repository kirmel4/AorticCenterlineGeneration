git clone https://github.com/vmtk/vmtk.git
git reset --hard c46de512701a072a8f453a045ce575b94cd3ba19
mkdir /vmtk-build && cd /vmtk-build && BUILD_SHARED_LIBS=OFF cmake /vmtk -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} -DPYTHON_LIBRARY=${PYTHON_LIBRARY} -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR}

cd /vmtk-build && make VMTK
