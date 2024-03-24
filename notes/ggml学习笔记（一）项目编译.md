# ggml学习笔记（一）项目编译

## 项目简介

核心代码如下：

```
.
├── CMakeLists.txt
├── LICENSE
├── README.md
├── ci
│   └── run.sh
├── cmake
│   ├── BuildTypes.cmake
│   └── GitVars.cmake
├── ggml.pc.in
├── include
│   └── ggml
│       ├── ggml-alloc.h
│       ├── ggml-backend.h
│       └── ggml.h
│
├── requirements.txt
└── src
    ├── CMakeLists.txt
    ├── ggml-alloc.c
    ├── ggml-backend-impl.h
    ├── ggml-backend.c
    ├── ggml-common.h
    ├── ggml-cuda.cu
    ├── ggml-cuda.h
    ├── ggml-impl.h
    ├── ggml-kompute.cpp
    ├── ggml-kompute.h
    ├── ggml-metal.h
    ├── ggml-metal.m
    ├── ggml-metal.metal
    ├── ggml-opencl.cpp
    ├── ggml-opencl.h
    ├── ggml-quants.c
    ├── ggml-quants.h
    ├── ggml-sycl.cpp
    ├── ggml-sycl.h
    ├── ggml-vulkan.cpp
    ├── ggml-vulkan.h
    └── ggml.c



```

可以看到ggml支持cpu、opencl、cuda、metal后端，ggml目前还没有多线程支持。

### 编译宏介绍

| 宏名                        | 宏说明                                                       |
| --------------------------- | ------------------------------------------------------------ |
| GGML_ALL_WARNINGS           | 是否开启编译的警告信息，默认`ON`                             |
| GGML_ALL_WARNINGS_3RD_PARTY | 是否开启第三方库的警告信息，默认`OFF`                        |
| GGML_SANITIZE_THREAD        | 是否开启线程检查工具，默认`OFF`                              |
| GGML_SANITIZE_ADDRESS       | 是否开启内存检查工具，默认`OFF`                              |
| GGML_SANITIZE_UNDEFINED     | 是否对未定义行为检测，默认`OFF`                              |
| GGML_BUILD_TESTS            | 构建测试，读取环境变量值`GGML_STANDALONE`，如果项目的根源目录（`CMAKE_SOURCE_DIR`）等于当前处理的CMake文件所在的目录（`CMAKE_CURRENT_SOURCE_DIR`），`GGML_STANDALONE`为`ON` |
| GGML_BUILD_EXAMPLES         | 同上                                                         |
| GGML_TEST_COVERAGE          | 使用测试用例，默认`OFF`                                      |
| GGML_PERF                   | 是否使用pert性能分析工具，用于查看cpu、内存等指标，默认`OFF` |
| GGML_NO_ACCELERATE          | 禁用加速框架，默认`OFF`                                      |
| GGML_OPENBLAS               | 是否使用高性能OpenBLAS线性代数库，OpenBLAS 在许多科学计算和数据分析库中被广泛使用，比如NumPy、SciPy等。默认`OFF` |
| GGML_CLBLAST                | 是否使用高性能线性代数CLBLAST库，专门用于在OpenCL（Open Computing Language）环境中进行高性能的基本线性代数操作。默认`OFF` |
| GGML_CUBLAS                 | 是 NVIDIA CUDA 平台上的一个高性能线性代数库。默认`OFF`       |
| GGML_METAL                  | 是否使用苹果的metal后端，默认`OFF`                           |
| GGML_AVX                    | 使用avx指令集优化，它引入了更宽的128位向量寄存器，允许单个指令同时处理更多的数据元素，从而提高并行计算能力。，默认`ON` |
| GGML_AVX2                   | 使用avx2指令集优化，AVX2 是对 AVX 的扩展，引入于Haswell微架构。它进一步增强了向量化能力。默认`ON` |
| GGML_AVX512                 | 使用avx512指令集优化，引入了更宽的512位向量寄存器，从而允许处理更多的数据元素同时进行高级的并行计算，只支持特定的英特尔处理器，如 Skylake、Cannon Lake、Ice Lake 等。默认`OFF` |
| GGML_AVX512_VBMI            | 使用AVX512_VBMI指令集优化，AVX-512 VBMI" 是指 AVX-512 指令集中的 Vector Byte Manipulation Instructions 扩展。VBMI 是 AVX-512 指令集的一部分，它引入了一组用于高效处理字节操作的指令，旨在加速字节级别的数据处理。默认`OFF` |
| GGML_AVX512_VNNI            | 使用GGML_AVX512_VNNI指令集优化，AVX-512 VNNI（Vector Neural Network Instructions）是英特尔处理器上的 AVX-512 指令集中的一个重要扩展，专门用于加速神经网络计算，特别是卷积神经网络（CNN）的操作。AVX-512 VNNI 扩展通常仅在支持该指令集的特定英特尔处理器上可用。这些处理器通常是为数据中心和高性能计算任务设计的。默认`OFF` |
| GGML_FMA                    | 是否开启FMA，FMA代表“Fused Multiply-Add”，它是一种在单个操作中执行浮点数乘法和加法的硬件指令，通过将乘法和加法结合在一起，FMA指令在一条指令周期内执行多个操作，从而显著提高了计算效率。默认`ON` |
| GGML_F16C                   | 使用F16C指令集优化，F16C 是一种英特尔处理器的指令集扩展，它主要用于在硬件级别上支持半精度（16位）浮点数操作。这种指令集扩展的目标是提供一种更高效地处理低精度浮点数的方式，以在一些应用中获得更好的性能和能效。仅在AVX2和AVX512指令集下默认`ON` |



## 源码编译

目前只支持linux和winsow，因为cmake里面只写了gnucc和clang编译器代码部分
环境要求：

* cmake >= 3.0
* gcc >= 4.9

```
mkdir build && cd build && cmake .. && make -j8
```

###  本人遇到的问题

1. 遇到的问题: fatal error: stdatomic.h: No such file or directory

​	原因：gcc版本过低,  需要升级gcc



2. 遇到问题： 在aarch64非苹果平台编译报错，selected processor does not support `fadd h0,h1,h0'

   原因：暂不支持该平台，