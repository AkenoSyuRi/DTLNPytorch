project(DTLN_deploy)
cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)
# for 1126
set(CMAKE_C_COMPILER "/home/lizhinan/project/lightse/ncnn/armv7toolchain_1126/gcc-linaro-6.3.1-2017.05-x86_64_arm-linux-gnueabihf/bin/arm-linux-gnueabihf-gcc")
set(CMAKE_CXX_COMPILER "/home/lizhinan/project/lightse/ncnn/armv7toolchain_1126/gcc-linaro-6.3.1-2017.05-x86_64_arm-linux-gnueabihf/bin/arm-linux-gnueabihf-g++")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fopenmp")

# for 1106
#set(CMAKE_C_COMPILER "/home/lizhinan/project/lightse/ncnn/armv7toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc")
#set(CMAKE_CXX_COMPILER "/home/lizhinan/project/lightse/ncnn/armv7toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-g++")

# for linux_ubuntu18.04
# fftw3
#set(FFTW3_LIBRARIES /home/lizhinan/project/lightse/DTLN_pytorch/deploy/third_party/fftw3/lib/libfftw3f.a)
#include_directories(/home/lizhinan/project/lightse/DTLN_pytorch/deploy/third_party/fftw3/include)

# ncnn
#set(NCNN_LIBS /home/lizhinan/project/lightse/DTLNPytorch/deploy/depend/ncnn_linux/lib/libncnn.a)
#include_directories(/home/lizhinan/project/lightse/DTLNPytorch/deploy/depend/ncnn_linux/include)

# for linux_1126
# FFTW3
set(FFTW3_LIBRARIES /home/lizhinan/project/lightse/DTLN_pytorch/deploy/third_party/fftw3_arm32_1126/lib/libfftw3f.a)
include_directories(/home/lizhinan/project/lightse/DTLN_pytorch/deploy/third_party/fftw3_arm32_1126/include/)

#ncnn
set(NCNN_LIBS /home/lizhinan/project/lightse/ncnn/build-arm-linux-gnueabihf/install/lib/libncnn.a)
include_directories(/home/lizhinan/project/lightse/ncnn/build-arm-linux-gnueabihf/install/include/)

#for linux_1106
#set(NCNN_LIBS /home/lizhinan/project/lightse/ncnn/build-arm-linux-gnueabihf/install/lib/libncnn.a)
#include_directories(/home/lizhinan/project/lightse/ncnn/build-arm-linux-gnueabihf/install/include/)

#set(FFTW3_LIBRARIES /home/lizhinan/project/lightse/DTLN_pytorch/deploy/third_party/fftw3_arm32/lib/libfftw3f.a)
#include_directories(/home/lizhinan/project/lightse/DTLN_pytorch/deploy/third_party/fftw3_arm32/include/)

FILE(GLOB SRC
        src/main_DTLN_NS.cpp
        src/fea_utils.cpp
        src/DTLN_NS_ncnn.cpp)




add_executable(dtln  ${SRC})
target_link_libraries(dtln ${FFTW3_LIBRARIES} ${NCNN_LIBS})

