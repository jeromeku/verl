#!/bin/bash
current_file=`realpath $0`
current_dir=`dirname ${current_file}`

export CUDNN_ROOT=${current_dir}/.venv/lib/python3.12/site-packages/nvidia/cudnn
export CUDNN_PATH=${CUDNN_ROOT}
export CPATH=${CUDNN_ROOT}/include:$CPATH
export LIBRARY_PATH=${CUDNN_ROOT}/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=${CUDNN_ROOT}/lib:$LD_LIBRARY_PATH
export CMAKE_CUDA_ARCHITECTURES=90
export NVTE_CUDA_ARCHS=90
export NVTE_CMAKE_EXTRA_ARGS="-DCMAKE_VERBOSE_MAKEFILE=1 -DCMAKE_EXPORT_COMPILE_COMMANDS=1"
export NVTE_FRAMEWORK=pytorch

TE_DIR=thirdparty/transformerengine
BUILD_DIR=${TE_DIR}/build
[[ -d "${BUILD_DIR}" ]] && rm -rf "${BUILD_DIR}"

uv pip install --no-build-isolation --editable ${TE_DIR} -v  2>&1 | tee _te_install.log