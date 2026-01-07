git clone https://github.com/ggml-org/llama.cpp.git
cd ./llama.cpp || (echo "Failed to enter to llama cpp directory" && exit)
cmake -B build \
-DGGML_CUDA=ON \
-DCMAKE_CUDA_ARCHITECTURES="86;120a-real" \
-DGGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
-DCMAKE_C_COMPILER=/usr/bin/gcc-12 \
-DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-12 \
-DCMAKE_BUILD_TYPE=Debug \
-DGGML_CCACHE=OFF
cmake --build build --config Debug -j $(nproc)
mkdir -p ../llama-lib
cp build/bin/* ../llama-lib/
cd ..
rm -rf ./llama.cpp