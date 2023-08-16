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



深度学习的推理框架有很多，现在主流的推理框架包括：[TensorRT](https://developer.nvidia.com/tensorrt)，[ONNXRuntime](https://onnxruntime.ai/)，[OpenVINO](https://link.zhihu.com/?target=https%3A//www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)，[ncnn](https://link.zhihu.com/?target=https%3A//github.com/Tencent/ncnn)，[MNN](https://link.zhihu.com/?target=https%3A//github.com/alibaba/MNN) 等。在知乎上也有关于深度学习推理框架的讨论[如何选择深度学习推理框架](https://www.zhihu.com/question/346965029)。

| 框架        | 算子数                                                       | 平台       |
| ----------- | ------------------------------------------------------------ | ---------- |
| onnxruntime | 164个算子                                                    | 多平台     |
| mnn         | 支持 178 个Tensorflow Op、52个 Caffe Op、163个 Torchscipts Op、158 个 ONNX Op | 多平台     |
| TensorRT    | 大部分onnx算子，可直接加载onnx格式                           | N卡        |
| OpenVINO    | 180个算子，可直接加载onnx格式                                | intel的cpu |
| ggml        | 73个算子                                                     | 多平台     |



本着调研与学习的目的，选择了较为简单的ggml作为入门框架






[ggml学习笔记（一）项目编译.md](notes/ggml学习笔记（一）项目编译.md)
