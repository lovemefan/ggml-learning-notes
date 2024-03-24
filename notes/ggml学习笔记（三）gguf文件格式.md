## GGUF是什么

> ***原文来自 [gguf.md](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)***

GGUF (GGML Universal File) 是用于存储 GGML 模型的二进制文件格式，旨在快速加载和保存模型。大多情况下模型都是使用 PyTorch 训练得到的，因此需要将pytorch的checkpoint转换为 GGUF 以在 GGML 中使用，gguf格式文件通常使用python库[gguf](https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/README.md)写入，c++读取。

GGUF 是 GGML、GGMF 和 GGJT 的后继文件格式，旨在通过保存加载模型所需的所有信息来确保无歧义。并且它还被设计为可扩展的，因此可以向模型添加新信息而不会破坏与现有模型的兼容性。



GGUF 是基于现有 GGJT 的格式，但对格式进行了一些更改，以使其更易于扩展和使用：

- 单文件部署：它们可以轻松分发和加载，并且不需要任何外部文件提供额外信息(因为gguf文件打包了超参数，词表，模型权重)。
- 可扩展：可以向基于 GGML 的代码添加新功能/向 GGUF 模型添加新信息，而不会破坏与现有模型的兼容性（原来GGJT格式的模型一旦被修改就可能会导致代码报错，不兼容）。
- `mmap` 兼容性：可以使用 `mmap` 快速加载和保存模型。
- 易于使用：可以使用少量代码轻松加载和保存模型，而无论使用的语言是什么，都不需要外部库。
- 完整信息：加载模型所需的所有信息都包含在模型文件中，用户不需要提供任何其他信息。

GGJT 和 GGUF 之间的主要区别是超参数（现在称为元数据）使用键值结构，而不是无类型值列表。这允许添加新的元数据而不会破坏与现有模型的兼容性，并且可以使用附加信息注释模型，这些信息对于推理或标识模型可能会很有用。

### 文件结构

**gguf文件保存了模型的超参、词表、模型权重**，gguf主要分为三个部分：

- 文件头
- 元数据的键值对，可用于保存超参，词表
- 模型权重的键值对

![](https://github.com/ggerganov/ggml/assets/1991296/c3623641-3a1d-408e-bfaf-1b7c4e16aa63)

*图示由 [@mishig25](https://github.com/mishig25)（GGUF v3）*



GGUF 文件的结构如上图。它们使用在下面称为 `ALIGNMENT` 的 `general.alignment` 元数据字段中指定的全局对齐。在需要时，文件用 `0x00` 字节填充到下一个 `general.alignment` 的倍数。

字段，包括数组，是按顺序写入的，除非另有说明。

模型默认是小端的。它们也可以以大端形式提供，以与大端计算机一起使用；在这种情况下，所有值（包括元数据值和张量）也将是大端的。在撰写本文时，没有办法确定模型是否为大端；这可能在未来版本中得到解决。如果未提供其他信息，则假定模型是小端的。



```c++
enum ggml_type: uint32_t {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 (5) support has been removed
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    // k-quantizations
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_I8,
    GGML_TYPE_I16,
    GGML_TYPE_I32,
    GGML_TYPE_COUNT,
};

enum gguf_metadata_value_type: uint32_t {
    // The value is a 8-bit unsigned integer.
    GGUF_METADATA_VALUE_TYPE_UINT8 = 0,
    // The value is a 8-bit signed integer.
    GGUF_METADATA_VALUE_TYPE_INT8 = 1,
    // The value is a 16-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT16 = 2,
    // The value is a 16-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT16 = 3,
    // The value is a 32-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT32 = 4,
    // The value is a 32-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT32 = 5,
    // The value is a 32-bit IEEE754 floating point number.
    GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    GGUF_METADATA_VALUE_TYPE_BOOL = 7,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    GGUF_METADATA_VALUE_TYPE_STRING = 8,
    // The value is an array of other values, with the length and type prepended.
    ///
    // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    GGUF_METADATA_VALUE_TYPE_ARRAY = 9,
    // The value is a 64-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT64 = 10,
    // The value is a 64-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT64 = 11,
    // The value is a 64-bit IEEE754 floating point number.
    GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
}

// A string in GGUF.
struct gguf_string_t {
    // The length of the string, in bytes.
    uint64_t len;
    // The string as a UTF-8 non-null-terminated string.
    char string[len];
}

union gguf_metadata_value_t {
    uint8_t uint8;
    int8_t int8;
    uint16_t uint16;
    int16_t int16;
    uint32_t uint32;
    int32_t int32;
    float float32;
    uint64_t uint64;
    int64_t int64;
    double float64;
    bool bool_;
    gguf_string_t string;
    struct {
        // Any value type is valid, including arrays.
        gguf_metadata_value_type type;
        // Number of elements, not bytes
        uint64_t len;
        // The array of values.
        gguf_metadata_value_t array[len];
    } array;
};

struct gguf_metadata_kv_t {
    // The key of the metadata. It is a standard GGUF string, with the following caveats:
    // - It must be a valid ASCII string.
    // - It must be a hierarchical key, where each segment is `lower_snake_case` and separated by a `.`.
    // - It must be at most 2^16-1/65535 bytes long.
    // Any keys that do not follow these rules are invalid.
    gguf_string_t key;

    // The type of the value.
    // Must be one of the `gguf_metadata_value_type` values.
    gguf_metadata_value_type value_type;
    // The value.
    gguf_metadata_value_t value;
};

struct gguf_header_t {
    // Magic number to announce that this is a GGUF file.
    // Must be `GGUF` at the byte level: `0x47` `0x47` `0x55` `0x46`.
    // Your executor might do little-endian byte order, so it might be
    // check for 0x46554747 and letting the endianness cancel out.
    // Consider being *very* explicit about the byte order here.
    uint32_t magic;
    // The version of the format implemented.
    // Must be `3` for version described in this spec, which introduces big-endian support.
    //
    // This version should only be increased for structural changes to the format.
    // Changes that do not affect the structure of the file should instead update the metadata
    // to signify the change.
    uint32_t version;
    // The number of tensors in the file.
    // This is explicit, instead of being included in the metadata, to ensure it is always present
    // for loading the tensors.
    uint64_t tensor_count;
    // The number of metadata key-value pairs.
    uint64_t metadata_kv_count;
    // The metadata key-value pairs.
    gguf_metadata_kv_t metadata_kv[metadata_kv_count];
};

uint64_t align_offset(uint64_t offset) {
    return offset + (ALIGNMENT - (offset % ALIGNMENT)) % ALIGNMENT;
}

struct gguf_tensor_info_t {
    // The name of the tensor. It is a standard GGUF string, with the caveat that
    // it must be at most 64 bytes long.
    gguf_string_t name;
    // The number of dimensions in the tensor.
    // Currently at most 4, but this may change in the future.
    uint32_t n_dimensions;
    // The dimensions of the tensor.
    uint64_t dimensions[n_dimensions];
    // The type of the tensor.
    ggml_type type;
    // The offset of the tensor's data in this file in bytes.
    //
    // This offset is relative to `tensor_data`, not to the start
    // of the file, to make it easier for writers to write the file.
    // Readers should consider exposing this offset relative to the
    // file to make it easier to read the data.
    //
    // Must be a multiple of `ALIGNMENT`. That is, `align_offset(offset) == offset`.
    uint64_t offset;
};

struct gguf_file_t {
    // The header of the file.
    gguf_header_t header;

    // Tensor infos, which can be used to locate the tensor data.
    gguf_tensor_info_t tensor_infos[header.tensor_count];

    // Padding to the nearest multiple of `ALIGNMENT`.
    //
    // That is, if `sizeof(header) + sizeof(tensor_infos)` is not a multiple of `ALIGNMENT`,
    // this padding is added to make it so.
    //
    // This can be calculated as `align_offset(position) - position`, where `position` is
    // the position of the end of `tensor_infos` (i.e. `sizeof(header) + sizeof(tensor_infos)`).
    uint8_t _padding[];

    // Tensor data.
    //
    // This is arbitrary binary data corresponding to the weights of the model. This data should be close
    // or identical to the data in the original model file, but may be different due to quantization or
    // other optimizations for inference. Any such deviations should be recorded in the metadata or as
    // part of the architecture definition.
    //
    // Each tensor's data must be stored within this array, and located through its `tensor_infos` entry.
    // The offset of each tensor's data must be a multiple of `ALIGNMENT`, and the space between tensors
    // should be padded to `ALIGNMENT` bytes.
    uint8_t tensor_data[];
};
```



## 如何自定义gguf文件

官方已经提供了一个python库 [gguf](https://github.com/ggerganov/llama.cpp/tree/master/gguf-py)用于写入gguf文件,安装如下

```bash
pip install gguf
```

官方还提供了一个转换huggingface模型的脚本[convert-hf-to-gguf.py](https://github.com/ggerganov/llama.cpp/blob/master/convert-hf-to-gguf.py), 以供参考。如果想更加灵活的使用gguf，可以参考一下[源码](https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/gguf/gguf_writer.py)。



以下给出一个简单的案例：

```python
from gguf import GGUFWriter 

def main():
    gguf_writer = GGUFWriter("example.gguf", "llama")

    gguf_writer.add_architecture() # 模型架构为GGUFWriter的第二个参数，即`llama`
    gguf_writer.add_block_count(12) # llama模型的模块数，专为LLM设计
    gguf_writer.add_uint32("answer", 42)  # 添加一个32-bit integer 类型的超参
    gguf_writer.add_float32("answer_in_float", 42.0)  # 添加一个32-bit float类型的超参
    gguf_writer.add_custom_alignment(64) # 设置自定义对齐值为64

    tensor1 = np.ones((32,), dtype=np.float32) * 100.0 
    tensor2 = np.ones((64,), dtype=np.float32) * 101.0
    tensor3 = np.ones((96,), dtype=np.float32) * 102.0

    gguf_writer.add_tensor("tensor1", tensor1) # 添加一个模型的tensor权重
    gguf_writer.add_tensor("tensor2", tensor2)
    gguf_writer.add_tensor("tensor3", tensor3)

    gguf_writer.write_header_to_file() # 写入头文件
    gguf_writer.write_kv_data_to_file() # 写入所有的键值对值， 包括超参等
    gguf_writer.write_tensors_to_file() # 写入所有的tensor 权重

    gguf_writer.close()
```



更多进阶用法可参考[源码](https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/gguf/gguf_writer.py)，该库添加了大量为LLM专门设计的键值对，如果想要更加灵活的设计gguf文件格式

只需要使用类似gguf_writer.add_uint32等更底层的函数即可，以下列出一些常用的函数就能满足各种超参个词表的定义：


- add_uint8(self, key: str, val: int) -> None
- add_int8(self, key: str, val: int) -> None
- add_uint16(self, key: str, val: int) -> None
- add_int16(self, key: str, val: int) -> None
- add_uint32(self, key: str, val: int) -> None
- add_int32(self, key: str, val: int) -> None
- add_float32(self, key: str, val: float) -> None
- add_uint64(self, key: str, val: int) -> None
- add_int64(self, key: str, val: int) -> None
- add_float64(self, key: str, val: float) -> None
- add_bool(self, key: str, val: bool) -> None
- add_string(self, key: str, val: str) -> None
- add_array(self, key: str, val: Sequence[Any]) -> None

