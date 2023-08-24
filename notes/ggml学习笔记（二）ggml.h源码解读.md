## 项目简介

ggml.h 包含了一系列的张量的算子操作、自动微分、以及基础的优化算法



### 宏介绍

| 宏名                    | 宏说明                                 |
| ----------------------- | -------------------------------------- |
| GGML_FILE_MAGIC         | ggml文件格式的magic？，值`0x67676d6c`  |
| GGML_FILE_VERSION       | ggml文件格式版本号，值`1`              |
| GGML_QNT_VERSION        | ggml量化版本号，值`1`                  |
| GGML_QNT_VERSION_FACTOR | ggml量化参数，暂时不是怎么用，值`1000` |
| GGML_MAX_DIMS           | ggml最大支持张量的维度，值`4`          |
| GGML_MAX_NODES          | ggml静态图支持的最大节点数，值`4096`   |
| GGML_MAX_PARAMS         | ggml支持的最大参数，值`256`            |
| GGML_MAX_CONTEXTS       | ，值`64`                               |
| GGML_MAX_SRC            | ，值`6`                                |
| GGML_MAX_NAME           | ，值`48`                               |
| GGML_MAX_OP_PARAMS      | ，值`32`                               |
| GGML_DEFAULT_N_THREADS  | 默认线程数，值`4`                      |
| GGML_EXIT_SUCCESS       | 退出成功                               |
| GGML_EXIT_ABORTED       | 退出中断                               |
| GGML_UNUSED(x)          |                                        |
| GGML_PAD(x, n)          | 用于内存对齐, 按照n字节对齐            |
| GGML_ASSERT(x)          |                                        |
| GGML_TENSOR_LOCALS_1    |                                        |
| GGML_TENSOR_LOCALS_2    |                                        |
| GGML_TENSOR_LOCALS_3    |                                        |
| GGML_TENSOR_LOCALS      |                                        |

### 枚举

| 枚举名           | 枚举说明                                                     |
| ---------------- | ------------------------------------------------------------ |
| ggml_type        | 用于枚举支持的数据类型，例如`GGML_TYPE_F32`,`GGML_TYPE_F16`  |
| ggml_backend     | 用于枚举支持的后端类型，例如`GGML_BACKEND_CPU`,`GGML_BACKEND_GPU` |
| ggml_ftype       | 用于枚举ggml支持的文件类型                                   |
| ggml_op          | 用于枚举所有支持的算子,例如 `GGML_OP_ARGMAX`                 |
| ggml_unary_op    | 用于枚举所有的一元运算符，例如`GGML_UNARY_OP_ABS`            |
| ggml_object_type | 用于枚举ggml对象类型，`ggml_object_type`, `GGML_OBJECT_GRAPH`, `GGML_OBJECT_WORK_BUFFER` |
|                  |                                                              |

### 结构体

| 结构体名                    | 结构体说明                                 |
| ----------------------- | -------------------------------------- |
| ggml_object | ggml对象，包含ggml对象和ggml_type |
| ggml_tensor | ggml张量 |
| ggml_cplan |  |
| ggml_cgraph |  |
| ggml_scratch |  |
| ggml_init_params |  |
| ggml_task_type |  |
| ggml_compute_params |  |



###  函数签名

| 函数名                                                       | 函数说明                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| void    ggml_time_init(void)                                 | 初始化时间                                                   |
| int64_t ggml_time_ms(void)                                   | 获取当前时间戳，单位为毫秒                                   |
| int64_t ggml_time_us(void)                                   | 获取当前时间戳，单位为微秒                                   |
| int64_t ggml_cycles(void);                                   |                                                              |
| int64_t ggml_cycles_per_ms(void)                             |                                                              |
| void ggml_numa_init(void)                                    | 初始化NUMA（Non-Uniform Memory Access, 非一致性内存访问），为了提升性能 |
| void ggml_print_object (const struct ggml_object * obj)      | 打印object                                                   |
| void ggml_print_objects(const struct ggml_context * ctx)     | 打印很多和object                                             |
| int64_t ggml_nelements   (const struct ggml_tensor * tensor) |                                                              |
| int64_t ggml_nrows       (const struct ggml_tensor * tensor) |                                                              |
| size_t  ggml_nbytes      (const struct ggml_tensor * tensor) |                                                              |
| ggml_nbytes_split(const struct ggml_tensor * tensor, int nrows_split) |                                                              |
| int ggml_blck_size (enum ggml_type type)                     |                                                              |
| size_t  ggml_type_size (enum ggml_type type)                 |                                                              |
| float   ggml_type_sizef(enum ggml_type type)                 |                                                              |
| const char * ggml_type_name(enum ggml_type type)             |                                                              |
| const char * ggml_op_name  (enum ggml_op  op)                |                                                              |
| const char * ggml_op_symbol(enum ggml_op   op)               |                                                              |
| size_t  ggml_element_size(const struct ggml_tensor * tensor) |                                                              |
| bool  ggml_is_quantized(enum ggml_type type)                 |                                                              |
|                                                              |                                                              |
|                                                              |                                                              |
|                                                              |                                                              |
|                                                              |                                                              |
|                                                              |                                                              |
|                                                              |                                                              |
|                                                              |                                                              |
|                                                              |                                                              |
|                                                              |                                                              |
|                                                              |                                                              |
|                                                              |                                                              |
|                                                              |                                                              |
|                                                              |                                                              |
|                                                              |                                                              |
|                                                              |                                                              |
|                                                              |                                                              |
|                                                              |                                                              |
|                                                              |                                                              |
|                                                              |                                                              |
|                                                              |                                                              |
|                                                              |                                                              |
|                                                              |                                                              |
|                                                              |                                                              |
|                                                              |                                                              |
|                                                              |                                                              |
|                                                              |                                                              |
|                                                              |                                                              |
|                                                              |                                                              |
|                                                              |                                                              |

