## GGML

> [**ggml**](https://github.com/ggerganov/ggml) is a tensor library for machine learning to enable large models and high performance on commodity hardware. It is used by [llama.cpp](https://github.com/ggerganov/llama.cpp) and [whisper.cpp](https://github.com/ggerganov/whisper.cpp)


ggml是用c语言实现的一套张量操作库， 官网在[这里](https://ggml.ai)。当前有以下优点

- 纯c实现
- 支持16bit浮点数
- 支持整数量化 (比如4-bit, 5-bit, 8-bit)
- 可以自动微分
- 内置优化算法 (例如 ADAM, L-BFGS)
- 苹果m系列芯片优化
- x86平台 AVX / AVX2 指令集优化
- 通过WebAssembly和 WASM SIMD技术可以跑在浏览器中
- 没有第三方库依赖
- 运行时不在新分配内存

现支持的后端有
- CPU
- CUDA
- Metal
- OpenCL
- Vulkan
- SYCL
- Kompute


深度学习的推理框架有很多，现在主流的推理框架包括：[TensorRT](https://developer.nvidia.com/tensorrt)，[ONNXRuntime](https://onnxruntime.ai/)，[OpenVINO](https://link.zhihu.com/?target=https%3A//www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)，[ncnn](https://link.zhihu.com/?target=https%3A//github.com/Tencent/ncnn)，[MNN](https://link.zhihu.com/?target=https%3A//github.com/alibaba/MNN) 等。在知乎上也有关于深度学习推理框架的讨论[如何选择深度学习推理框架](https://www.zhihu.com/question/346965029)。

| 框架          | 算子数                                                                  | 平台        |
|-------------|----------------------------------------------------------------------|-----------|
| onnxruntime | 164个算子                                                               | 多平台       |
| ncnn        | 约197个算子                                                              | 多平台，主打移动端 |
| mnn         | 支持 178 个Tensorflow Op、52个 Caffe Op、163个 Torchscipts Op、158 个 ONNX Op | 多平台       |
| TensorRT    | 大部分onnx算子，可直接加载onnx格式                                                | N卡        |
| OpenVINO    | 180个算子，可直接加载onnx格式                                                   | intel的cpu |
| ggml        | 73个算子                                                                | 多平台       |


国内也有着好几个非常优秀的推理框架，例如腾讯的ncnn起步早且社区活跃，支持平台广泛，主打移动端部署，还有阿里的mnn，都非常优秀。

本着调研与学习的目的，这里选择了最简单的ggml作为入门学习框架， ggml也是作为了whisper.cpp和llama.cpp的后端推理框架。
目前为止，当前该库仍在开发中，几个月不见又增加很多新的工作，废弃了一些接口。 
并且ggml的官方文档还是一个TODO状态，
本项目的出新在于学习当前一些主流的推理框架，为后续深度学习模型落地做准备。





[ggml学习笔记（一）项目编译.md](notes/ggml学习笔记（一）项目编译.md)

[ggml学习笔记（二）ggml.h源码解读.md](notes/ggml学习笔记（二）ggml.h源码解读.md)
