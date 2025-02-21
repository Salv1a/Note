# PyTorch动态调度

**Author: Sheng Wang **

**Date: 24-11-18**



***以下为gpt的问答，待整理***



pytorch中aten算子的调用问题， aten中nativate function.yaml中有以下两个func

```yaml
- func: masked_fill_.Scalar(Tensor(a!) self, Tensor mask, Scalar value) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  variants: method
  dispatch:
    CPU: masked_fill__cpu
    CUDA: masked_fill__cuda
    QuantizedCPU: masked_fill__quantized_cpu
    QuantizedCUDA: masked_fill__quantized_cuda
    MPS: masked_fill__mps
  autogen: masked_fill.Scalar_out

- func: masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor
```

我在调用torch.masked_fill时，torch是如何选择调用哪一个的



###### ChatGPT 说：

ChatGPT

在 PyTorch 中，`torch.masked_fill` 的调用逻辑与 `aten` 中的 `native_functions.yaml` 定义息息相关。PyTorch 会根据函数调用的上下文、张量的类型、设备等因素选择合适的实现。

### 分析两个 `masked_fill` 的定义

1. **`masked_fill_.Scalar`**

   - 带有下划线的 `masked_fill_` 是一个 **in-place 操作**。它会直接修改 `self` 张量，而不会返回新的张量。

   - 函数签名中 

     ```
     (Tensor(a!) self, Tensor mask, Scalar value) -> Tensor(a!)
     ```

      表示：

     - `self` 是一个输入输出张量（标记为 `a!`，表示可修改）。
     - 返回值还是修改后的 `self`。

   - 此函数通过 `variants: method` 注册为张量的方法。

2. **`masked_fill.Scalar`**

   - 这是一个 **out-of-place 操作**。它不会修改原来的张量，而是返回一个新的张量。

   - 函数签名 

     ```
     (Tensor self, Tensor mask, Scalar value) -> Tensor
     ```

      表示：

     - `self` 是输入张量，不可修改。
     - 返回值是一个新的张量。

### `torch.masked_fill` 的选择逻辑

- PyTorch 提供的 API `torch.masked_fill` 实际上会根据用户的调用方式和上下文，决定调用哪一个具体实现。
- 具体选择逻辑如下：

#### 1. 如果用户调用的是 **方法形式**：

```python
tensor.masked_fill(mask, value)
```

- PyTorch 会将其解析为张量的 **方法调用**，即 `masked_fill_` 或 `masked_fill`。
- 根据上下文：
  - 如果调用的是 `masked_fill_`，这将直接修改 `tensor`，对应 `masked_fill_.Scalar`。
  - 如果调用的是 `masked_fill`，这将返回一个新张量，对应 `masked_fill.Scalar`。

#### 2. 如果用户调用的是 **函数形式**：

```python
torch.masked_fill(tensor, mask, value)
```

- PyTorch 会调用 `masked_fill.Scalar`，即返回一个新张量的版本。
- 这是因为 `torch` API 默认不执行 in-place 操作，除非明确调用带下划线的方法（如 `masked_fill_`）。

### 如何选择后端实现

`native_functions.yaml` 中定义了不同设备的 `dispatch` 条目，例如 `CPU`, `CUDA`, `QuantizedCPU`，以及 `MPS` 等。PyTorch 根据以下条件选择合适的后端实现：

1. **设备选择**：
   - 如果 `tensor` 位于 CPU 上，调用 `masked_fill__cpu`。
   - 如果 `tensor` 位于 GPU 上（CUDA），调用 `masked_fill__cuda`。
   - 对于其他设备如 Quantized 和 MPS，PyTorch 会选择相应的实现。
2. **类型检查和匹配**：
   - PyTorch 会在运行时检查张量的类型和属性（是否量化、设备等），并基于 `dispatch` 表选择合适的后端实现。
3. **In-place 与 Out-of-place**：
   - 如果是 in-place 操作（`masked_fill_`），PyTorch 会调用 `masked_fill_.Scalar` 的对应实现。
   - 如果是 out-of-place 操作（`masked_fill`），PyTorch 会调用 `masked_fill.Scalar` 的对应实现。

### 总结

- `torch.masked_fill` 默认调用 `masked_fill.Scalar`，返回一个新的张量。
- `tensor.masked_fill_` 会调用 `masked_fill_.Scalar`，直接修改张量。
- 最终具体实现（如 `masked_fill__cpu` 或 `masked_fill__cuda`）由张量所在的设备和类型动态决定。

通过这种机制，PyTorch 提供了灵活的前端 API，同时保持高效的后端执行。





我在外部执行torch.masked_fill，是out-of-place操作，并使用library.impl重新注册了in-place操作masked_fill_的函数a，实际执行时，函数a会执行，但返回的结果是仍然是out-of-place的结果，为什么



###### ChatGPT 说：

ChatGPT

这种行为可能与 PyTorch 的 **dispatch 机制** 和 **函数注册体系** 有关。以下是导致这种情况的可能原因和解决方案。

------

### 原因分析

#### 1. **`torch.masked_fill` 和 `masked_fill_` 调用路径不同**

- 当你调用 `torch.masked_fill` 时，默认情况下，它会调用 `aten::masked_fill` 的 out-of-place 实现 (`masked_fill.Scalar`)。
- 即使你重新注册了 `masked_fill_` 的实现 (`library.impl`)，它并不会影响 `torch.masked_fill` 的行为，因为两者分别对应不同的函数。

#### 2. **`library.impl` 注册修改范围有限**

- `library.impl` 注册机制主要影响特定设备（例如 `CPU`, `CUDA`）和特定操作（如 `masked_fill_`）。
- 当你通过 `library.impl` 修改 `masked_fill_` 的行为时，只有显式调用 `masked_fill_`（in-place 操作）才会触发新的实现。
- `torch.masked_fill` 会忽略 `masked_fill_` 的实现，仍然调用 `masked_fill.Scalar`。

#### 3. **函数返回值未正确处理**

- 如果你的 `masked_fill_` 实现 (函数 `a`) 被调用后返回了错误的张量引用，那么 PyTorch 的调度可能在内部将结果替换为默认 out-of-place 的行为。
- 这种情况可能与 `TensorIterator` 或 `native::Tensor` 操作方式有关，具体取决于你如何实现 `a`。

------

### 解决方案

#### **确认函数注册的目标**

1. 确保你对 `torch.masked_fill` 和 `masked_fill_` 的注册操作没有混淆。

   - 对于 `masked_fill_`，需要确保注册的是 `aten::masked_fill_.Scalar`。
   - 对于 `torch.masked_fill`，需要明确它调用的是 `aten::masked_fill.Scalar`。

2. 如果你希望全局替换 `torch.masked_fill` 的行为，可以尝试用 `library.impl` 针对 `masked_fill.Scalar` 重新注册函数。

   ```python
   from torch.utils.cpp_extension import Library
   
   lib = Library('aten', 'IMPL')
   lib.impl('masked_fill.Scalar', my_function, 'CPU')  # 或 'CUDA'，视需求而定
   ```

#### **正确处理函数返回值**

- 如果你的函数 `a` 修改的是 in-place 张量（如 `self`），需要确保它正确返回被修改的 `self`。

- 如果实现中有返回值问题，可能导致 PyTorch 回退到默认的 out-of-place 行为。

  确保函数 `a` 的签名与 `masked_fill_.Scalar` 的要求一致，例如：

  ```python
  def a(self, mask, value):
      # 修改 self 的内容
      self[mask] = value
      return self  # 确保返回的是修改后的张量
  ```

#### **测试和验证**

- 使用 `torch.masked_fill_` 显式调用 in-place 操作，确保你的 `a` 实现生效。

- 如果仍有问题，可以通过以下调试函数验证实际调用的实现：

  ```python
  print(torch._C._dispatch_find_schema('aten::masked_fill_'))
  print(torch._C._dispatch_find_schema('aten::masked_fill'))
  ```

------

### 注意事项

- PyTorch 的 `dispatch` 机制动态选择函数实现，因此重新注册某些操作时需要特别注意对应的 `schema`。
- 如果你希望完全替换 `torch.masked_fill` 的行为，建议直接替换 `masked_fill.Scalar` 的实现，而不是 `masked_fill_`。





我的实现有问题，但产生的结果我难以理解，先忽略我实现的正确性。我实现了一个函数a去替换masked_fill_.Scalar，只注册了masked_fill_.Scalar，但函数中写错了，返回了一个新的张量而不是输入inp。实际调用时，我调用torch.masked_fill，默认应该执行out-of-place操作即masked_fill.Scalar，为什么函数a也被执行了，并且直接返回了函数a中未被修改的inp



###### ChatGPT 说：

ChatGPT

这个问题的核心在于 PyTorch 的 **动态调度机制** 和 **函数实现注册**。以下是具体分析以及原因。

------

### 1. **`torch.masked_fill` 的调度机制**

- 当调用 `torch.masked_fill` 时，它默认映射到 `masked_fill.Scalar`（out-of-place 实现）。
- 但如果你只注册了 `masked_fill_.Scalar`（in-place 实现），某些情况下，PyTorch 的内部调度会尝试回退到注册的 `masked_fill_.Scalar`，特别是当设备或后端的具体实现不可用或冲突时。

这种回退机制的逻辑可能会导致 PyTorch调用到你注册的 `masked_fill_.Scalar`（即函数 `a`）。

------

### 2. **错误的函数实现导致结果问题**

你提到 `a` 中返回了一个新的张量而不是修改输入张量 `inp`。根据 PyTorch 的约定：

- `masked_fill_.Scalar` 的签名要求它是 **in-place 操作**，必须返回输入张量（即被修改的 `inp`）。
- 如果你实现的函数返回了一个新的张量，PyTorch 的调度机制会尝试将这个返回值直接作为结果传递，导致行为异常。

------

### 3. **PyTorch 如何处理函数返回值**

- PyTorch 期望 `masked_fill_.Scalar` 返回修改后的输入张量 `inp`。但你的函数返回了一个新的张量，违反了这个约定。
- 由于 PyTorch 的内部实现可能会缓存或引用 `inp`，即使你的函数返回了一个新张量，PyTorch 可能仍然继续操作原始 `inp`。

这就可能导致以下情况：

- **函数 `a` 被执行**：因为 PyTorch 调用了 `masked_fill_.Scalar` 作为实现。
- **未被修改的 `inp` 被返回**：由于你的实现中返回了错误的张量，PyTorch 内部可能在引用 `inp` 时保留了它，导致返回了原始的 `inp`。

------

### 4. **为什么 `masked_fill.Scalar` 被映射到 `masked_fill_.Scalar`？**

- 在 `library.impl` 中，你覆盖了 `masked_fill_.Scalar`，而没有注册 `masked_fill.Scalar`。
- 如果 `masked_fill.Scalar` 在特定设备或后端中没有明确的实现，PyTorch 会尝试 fallback 到已有实现。
- 因为 `masked_fill_.Scalar` 是为同类操作注册的，PyTorch 的调度逻辑可能直接选择调用它。

------

### 5. **结论**

- **函数 `a` 被执行**：由于 PyTorch 的回退机制，当 `masked_fill.Scalar` 找不到实现时，它调用了 `masked_fill_.Scalar`（函数 `a`）。
- **返回未被修改的 `inp`**：这是因为你的 `a` 函数返回了一个新的张量，而 PyTorch 内部可能仍然使用 `inp` 作为返回值。

------

### 6. **如何验证和修复**

#### 验证：

- 打印调度过程，确认 

  ```python
  torch.masked_fill
  ```

   调用了哪个实现：

  ```python
  print(torch._C._dispatch_find_schema('aten::masked_fill'))
  print(torch._C._dispatch_find_schema('aten::masked_fill_'))
  ```

- 检查 `masked_fill.Scalar` 是否回退到 `masked_fill_.Scalar`。

#### 修复：

- 确保正确注册 `masked_fill.Scalar` 和 `masked_fill_.Scalar`，不要混用。

- 修改函数 `a`，确保符合 in-place 的约定：

  ```python
  def a(inp, mask, value):
      # 正确实现：in-place 修改 inp
      inp[mask] = value
      return inp  # 必须返回 inp
  ```

- 如果需要替换 `masked_fill.Scalar` 的实现：

  ```python
  lib.impl('masked_fill.Scalar', my_function, 'CPU')
  ```

这样可以避免因调度机制混淆而产生的问题。





请仔细介绍一下pytorch的回退机制和调度逻辑，我只注册了masked_fill_.Scalar，但torch 本身应该有masked_fill.Scalar的实现，为什么会调用到masked_fill_.Scalar



###### ChatGPT 说：

ChatGPT

PyTorch 的回退机制和调度逻辑是其动态分派（dispatch）系统的核心部分，用于根据输入张量的设备、类型和操作模式选择合适的实现。当行为和预期不一致时，通常是因为调度路径被修改、注册冲突或者错误实现引发的。以下是详细分析和解释。

------

## **PyTorch 的调度机制概述**

PyTorch 的运算调度基于 `ATen`，其核心是动态分派（dynamic dispatch）系统。调度的逻辑主要依赖以下几个因素：

1. **Operator Schema**：
   - 每个操作的定义（schema）明确规定了参数和返回值的类型。
   - 例如，`masked_fill.Scalar` 和 `masked_fill_.Scalar` 是不同的操作，分别对应 out-of-place 和 in-place 的行为。
2. **Backend/Device Dispatch**：
   - 操作根据张量的设备（CPU、CUDA、Quantized、MPS 等）选择对应的实现。
   - 示例：
     - `masked_fill.Scalar` 的 CPU 实现是 `masked_fill_cpu`。
     - `masked_fill_.Scalar` 的 CUDA 实现是 `masked_fill__cuda`。
3. **Variant Dispatch**：
   - 操作分为方法（method）和函数（function）两种调用方式：
     - 方法：`tensor.masked_fill_()` 对应 in-place 操作。
     - 函数：`torch.masked_fill()` 对应 out-of-place 操作。
4. **Fallback Logic**：
   - 当某个后端或操作的具体实现不可用时，PyTorch 会尝试回退到通用实现（如果存在）。
   - 例如，对于某些特殊设备（如 XLA），PyTorch 会尝试调用 CPU 版本作为替代。

------

## **为什么会调用到 `masked_fill_.Scalar`？**

### **1. 调度顺序**

PyTorch 在解析 `torch.masked_fill` 时，首先根据输入张量的设备和类型查找 `masked_fill.Scalar` 的实现。如果：

- **未找到适合的实现**：PyTorch 会尝试调用相似的实现（fallback）。
- **操作被错误注册或修改**：可能导致 `masked_fill.Scalar` 映射到了错误的实现。

由于你通过 `library.impl` 仅注册了 `masked_fill_.Scalar`，PyTorch 的调度系统可能错误地认为这是 `masked_fill.Scalar` 的一个兼容实现，从而将其映射到 `masked_fill_.Scalar`。

------

### **2. 调度路径的动态映射**

PyTorch 的调度路径是动态生成的，以下是 `torch.masked_fill` 的大致调用流程：

1. 根据操作名称 `masked_fill`，查找操作的注册表。
   - 如果 `masked_fill.Scalar` 在目标设备（如 CPU）上有明确的实现，则直接调用它。
   - 如果未找到，可能回退到其他相关实现，例如 `masked_fill_.Scalar`。
2. 如果 `masked_fill.Scalar` 和 `masked_fill_.Scalar` 共用某些实现路径（例如共享的内核或通用后端函数），注册错误可能会导致 PyTorch 调用 `masked_fill_.Scalar`。

------

### **3. 你注册的操作覆盖了原生实现**

通过 `library.impl` 注册 `masked_fill_.Scalar` 后，PyTorch 的调度系统可能优先使用你的注册函数，而忽略原始的 `masked_fill.Scalar` 实现。这可能是由于：

- PyTorch 的调度系统认为 `masked_fill_.Scalar` 能够处理类似的输入。
- 操作的设备特化实现（如 CPU、CUDA）被覆盖时，默认回退到用户注册的函数。

------

### **调度冲突的验证与排查**

你可以通过以下方法验证具体调度逻辑：

#### **1. 打印操作注册表**

使用 PyTorch 内部工具查看 `masked_fill` 和 `masked_fill_` 的注册信息：

```python
print(torch._C._dispatch_find_schema('aten::masked_fill'))
print(torch._C._dispatch_find_schema('aten::masked_fill_'))
```

- 这会返回 `masked_fill.Scalar` 和 `masked_fill_.Scalar` 的实际绑定函数。
- 检查是否 `masked_fill.Scalar` 被错误地映射到了 `masked_fill_.Scalar`。

#### **2. 检查具体实现的后端映射**

查看目标设备上的实现是否被覆盖：

```python
from torch.utils._pytree import tree_map
print(torch._C._dispatch_dump_registry('masked_fill'))
print(torch._C._dispatch_dump_registry('masked_fill_'))
```

------

## **如何修复或避免这种问题？**

### **1. 确保注册的操作不冲突**

- 如果你只想覆盖 `masked_fill_.Scalar` 的行为，避免影响 `masked_fill.Scalar`，需要精确指定目标后端和操作。

- 示例：注册 

  ```
  masked_fill_.Scalar
  ```

   的 CPU 实现，不影响其他后端：

  ```
  python
  
  
  复制代码
  from torch.utils.cpp_extension import Library
  
  lib = Library('aten', 'IMPL')
  lib.impl('masked_fill_.Scalar', my_function, 'CPU')  # 仅覆盖 CPU 的实现
  ```

### **2. 明确注册 `masked_fill.Scalar` 的实现**

- 如果你需要覆盖 

  `torch.masked_fill`的行为，确保直接注册 `masked_fill.Scalar`

  ```python
  lib.impl('masked_fill.Scalar', my_function, 'CPU')
  ```

### **3. 防止误注册全局实现**

- 避免不必要的全局注册。如果你注册了全局实现，例如：

  ```python
  lib.impl('masked_fill_.Scalar', my_function)
  ```

  这可能导致其他后端（如 CUDA）也尝试调用该实现，引发调度问题。

### **4. 检查返回值与规范一致性**

- `masked_fill_.Scalar` 的实现必须返回修改后的输入张量。如果返回值不符合规范，PyTorch 的动态调度可能会尝试 fallback 到其他实现，导致混乱行为。

------

## **总结**

- **原因**：你注册了 `masked_fill_.Scalar`，可能覆盖了部分设备的 `masked_fill.Scalar` 调用路径，导致调度回退到错误实现。
- **解决方案**：明确注册目标操作和设备，确保 `masked_fill.Scalar` 和 `masked_fill_.Scalar` 的实现逻辑独立，不冲突。
- **验证工具**：通过 PyTorch 的调度注册工具 (`_dispatch_find_schema` 和 `_dispatch_dump_registry`) 检查具体调用路径。

通过以上方法，可以确保调度逻辑符合预期，避免不必要的回退或覆盖行为。





实际pytorch的yaml中是这样写的，我的注册代码是

```yaml
lib.impl("masked_fill_.Scalar", masked_fill, "CUDA")
- func: masked_fill_.Scalar(Tensor(a!) self, Tensor mask, Scalar value) -> Tensor(a!)
  device_check: NoCheck   # TensorIterator
  variants: method
  dispatch:
    CPU: masked_fill__cpu
    CUDA: masked_fill__cuda
    QuantizedCPU: masked_fill__quantized_cpu
    QuantizedCUDA: masked_fill__quantized_cuda
    MPS: masked_fill__mps
  autogen: masked_fill.Scalar_out

- func: masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: masked_fill
    NestedTensorCPU, NestedTensorCUDA: NestedTensor_masked_fill
  tags: pointwise
```





###### ChatGPT 说：

ChatGPT

根据提供的信息，你通过 `lib.impl("masked_fill_.Scalar", masked_fill, "CUDA")` 注册了自定义的 `masked_fill_.Scalar` 实现，但 `torch.masked_fill`（对应 `masked_fill.Scalar` 的函数实现）意外调用了 `masked_fill_.Scalar`。这是由于 PyTorch 的动态调度机制和 `CompositeExplicitAutograd` 的 fallback 特性共同作用的结果。

以下是详细的分析和解释：

------

### **PyTorch 的动态调度与 `CompositeExplicitAutograd`**

1. **`masked_fill.Scalar` 的调度路径**
   - 从 YAML 可以看出，`masked_fill.Scalar`的后端实现分为以下两种情况：
     - **CompositeExplicitAutograd**：表示这是一个复合运算，使用由其他底层操作（如 `masked_fill_.Scalar`）组合实现，适用于大多数后端。
     - **NestedTensorCPU/NestedTensorCUDA**：特化实现，仅在 NestedTensor 上使用。
2. **`masked_fill_.Scalar` 的调度路径**
   - `masked_fill_.Scalar` 是一个独立的操作，明确支持各后端设备（CPU、CUDA、Quantized 等）。
   - 它是原生的低级内核操作，通常会被 `masked_fill.Scalar` 调用作为底层实现。
3. **为何 `masked_fill.Scalar` 调用了 `masked_fill_.Scalar`？**
   - `masked_fill.Scalar` 使用 `CompositeExplicitAutograd` 来表示，这意味着它通过高层封装实现，但最终可能依赖于底层的 `masked_fill_.Scalar`。
   - 当 `torch.masked_fill`被调用时：
     1. 如果是普通张量，`masked_fill.Scalar` 的 `CompositeExplicitAutograd` 会解析为对 `masked_fill_.Scalar` 的调用。
     2. 你注册了自定义的 `masked_fill_.Scalar`（仅限 CUDA），因此当调度到 `CUDA` 后端时，PyTorch 使用了你的实现。

------

### **关键问题：`masked_fill.Scalar` 调用了 `masked_fill_.Scalar` 的错误实现**

你注册的 `masked_fill_.Scalar` 实现可能不符合 in-place 操作的规范，但 PyTorch 的 `CompositeExplicitAutograd` 逻辑仍然使用了它。这导致以下现象：

1. **调用 `masked_fill.Scalar`**：
   - 按照 `CompositeExplicitAutograd` 的定义，它会间接调用 `masked_fill_.Scalar`。
   - 由于你覆盖了 CUDA 设备的 `masked_fill_.Scalar`，因此触发了你的自定义实现。
2. **返回值不符合预期**：
   - 你的实现可能返回了一个新张量，而不是 in-place 修改后的输入张量。
   - PyTorch 的高层逻辑期待 in-place 修改，但由于返回值不符合规范，可能保留了原始输入张量的引用。

------

### **如何验证行为？**

#### **检查 `masked_fill.Scalar` 的调度路径**

运行以下代码确认 `torch.masked_fill` 的实际实现：

```python
print(torch._C._dispatch_find_schema('aten::masked_fill'))
```

检查是否返回类似以下内容：

```scss
Schema(name='aten::masked_fill', overload_name='Scalar', arguments=[...])
DispatchKeySet({CompositeExplicitAutograd, NestedTensorCUDA})
```

#### **检查 `masked_fill_.Scalar` 的覆盖**

查看 CUDA 上的 `masked_fill_.Scalar` 是否被你的实现替换：

```python
print(torch._C._dispatch_find_schema('aten::masked_fill_'))
```

检查是否包含你的自定义实现。

#### **验证 `CompositeExplicitAutograd` 的调用逻辑**

在你的自定义 `masked_fill_.Scalar` 中添加调试输出，确认是否从 `masked_fill.Scalar` 路径调用：

```python
def masked_fill(input, mask, value):
    print("Custom masked_fill_.Scalar called")
    # Your implementation
```

------

### **如何修复问题？**

#### **1. 确保 `masked_fill_.Scalar` 的实现符合规范**

你实现的 `masked_fill_.Scalar` 应该符合以下要求：

- 修改 `self` 的内容并返回修改后的 `self`。
- 确保函数签名与 `masked_fill_.Scalar` 一致。

例如：

```python
def masked_fill(input, mask, value):
    # Ensure in-place behavior
    input[mask] = value
    return input  # Must return input
```

#### **2. 避免错误覆盖其他路径**

如果你仅希望自定义 in-place 操作，而不影响 `torch.masked_fill` 的行为，可以同时明确注册 `masked_fill.Scalar` 和 `masked_fill_.Scalar`：

```python
lib.impl("masked_fill.Scalar", your_out_of_place_function, "CUDA")
lib.impl("masked_fill_.Scalar", your_in_place_function, "CUDA")
```

#### **3. 排查其他冲突**

通过以下工具确保没有其他设备或后端的实现冲突：

```python
print(torch._C._dispatch_dump_registry("masked_fill"))
print(torch._C._dispatch_dump_registry("masked_fill_"))
```

------

### **总结**

- `torch.masked_fill` (out-of-place) 调用 `masked_fill.Scalar`，而后者可能通过 `CompositeExplicitAutograd` 依赖于 `masked_fill_.Scalar`。
- 由于你通过 `lib.impl` 覆盖了 `masked_fill_.Scalar`，你的自定义实现被用作底层操作。
- 问题的根源在于你的实现不符合 in-place 的规范，但被 `CompositeExplicitAutograd` 间接调用，导致不符合预期的行为。
- 修复方式是确保 `masked_fill_.Scalar` 返回符合规范的结果，并在需要时明确注册 `masked_fill.Scalar` 的实现，避免冲突。





pytorch的CompositeExplicitAutograd dispatch是什么，是如何作用的



###### ChatGPT 说：

ChatGPT

`CompositeExplicitAutograd` 是 PyTorch 动态调度系统中的一个 **dispatch key**，用于管理运算的实现方式。它的主要作用是作为一种 **复合运算** 的实现策略，通过已有的低级操作组合实现高层的算子逻辑。

以下是详细介绍 `CompositeExplicitAutograd` 的作用机制及其在 PyTorch 调度系统中的位置。

------

## **1. 什么是 CompositeExplicitAutograd？**

在 PyTorch 的调度系统中，每个操作都绑定到不同的 `dispatch key`。`CompositeExplicitAutograd` 是这些 `dispatch key` 中的一个特殊类型，专门用于：

- **复合操作**：它将高层操作实现为一系列低层原语（例如 `add`, `mul`, `masked_fill_` 等）。
- **支持 Autograd**：这些操作在 `CompositeExplicitAutograd` 中明确支持梯度计算，即它们是由一组支持 `Autograd` 的底层操作实现的。

相比之下，另一个类似的 `dispatch key` 是 `CompositeImplicitAutograd`，它仅在不需要梯度计算的情况下工作。

------

## **2. CompositeExplicitAutograd 的设计目的**

- **可移植性**：允许高层操作通过底层操作组合实现，无需为每种设备（如 CPU、CUDA）实现特化的内核。
- **代码复用**：避免重复实现特定设备上的逻辑，直接利用低级操作。
- **清晰的梯度支持**：确保这些操作的复合实现可以自动参与 Autograd。

### 复合操作的优点

- **简单性**：开发者无需为每个后端（CPU、CUDA 等）实现特化操作。
- **灵活性**：可以快速支持新设备（如 MPS 或 XLA），只需确保底层操作在目标设备上可用。

------

## **3. CompositeExplicitAutograd 是如何作用的？**

当 PyTorch 调度系统遇到一个绑定到 `CompositeExplicitAutograd` 的操作时：

### **1. 查找具体实现**

- PyTorch 检查操作是否有特化的实现（例如，CPU、CUDA）。
- 如果没有特化实现，它会回退到 `CompositeExplicitAutograd`，并使用复合实现。

### **2. 使用底层操作实现**

`CompositeExplicitAutograd` 将调用一组已经存在的低级操作实现高层功能。例如：

```yaml
- func: masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor
  dispatch:
    CompositeExplicitAutograd: masked_fill
```

这表明 `masked_fill.Scalar` 的逻辑是通过其他低级操作（如 `masked_fill_` 或 `where`）组合实现。

------

### **例子：masked_fill.Scalar**

假设 `masked_fill.Scalar` 的 `CompositeExplicitAutograd` 实现如下：

```python
def masked_fill(self, mask, value):
    return torch.where(mask, torch.full_like(self, value), self)
```

1. **作为复合操作**：
   - `masked_fill.Scalar` 不需要特化的 CUDA 或 CPU 实现。
   - 调度系统直接使用高层逻辑调用 `torch.where` 和 `torch.full_like` 实现其功能。
2. **与 Autograd 的集成**：
   - 因为底层操作（`torch.where`, `torch.full_like`）已经支持 Autograd，`masked_fill.Scalar` 的梯度计算也自然得到支持。

------

## **4. CompositeExplicitAutograd 的调度流程**

PyTorch 调度系统根据 `dispatch key` 查找适当的实现，以下是具体流程：

### 调度优先级

1. 检查是否有设备特化实现（如 CPU、CUDA）。
2. 检查是否有量化、元张量（meta-tensor）、稀疏（sparse）等特化实现。
3. 如果以上都不存在，回退到 `CompositeExplicitAutograd`。

### 示例：调用 `torch.masked_fill`

假设你调用以下代码：

```python
result = torch.masked_fill(tensor, mask, value)
```

#### 调度路径

1. PyTorch 首先检查是否有 `masked_fill.Scalar` 的设备特化实现。
2. 如果找不到（例如，MPS 后端未特化），则回退到 `CompositeExplicitAutograd`。
3. 使用高层逻辑（如 `torch.where`）调用底层操作实现。

------

## **5. 如何验证 CompositeExplicitAutograd 是否被调用？**

你可以通过调试工具检查调度路径：

```python
print(torch._C._dispatch_find_schema('aten::masked_fill'))
```

查看是否返回以下内容：

```scss
Schema(name='aten::masked_fill', overload_name='Scalar', arguments=[...])
DispatchKeySet({CompositeExplicitAutograd, ...})
```

这表明 `CompositeExplicitAutograd` 是此操作的备选实现。

------

## **6. CompositeExplicitAutograd 与用户自定义实现的交互**

当用户通过 `lib.impl` 注册自己的操作实现时，PyTorch 的调度可能会覆盖或混用 `CompositeExplicitAutograd`：

- **优先级问题**： 如果你覆盖了 `masked_fill_.Scalar`（in-place 操作），但 `CompositeExplicitAutograd` 调用了它作为底层操作，行为可能与预期不符。
- **正确性问题**： 如果你注册的操作不符合 `CompositeExplicitAutograd` 的要求（如 in-place 行为不正确），调度结果可能异常。

------

## **7. 何时避免 CompositeExplicitAutograd？**

- 如果设备（如 CUDA）有性能优化需求，可以通过注册特化实现覆盖 `CompositeExplicitAutograd`。
- 如果你需要完全控制某个操作的行为，可以明确覆盖目标设备的实现，而不仅限于复合逻辑。

例如：

```python
from torch.utils.cpp_extension import Library
lib = Library("aten", "IMPL")
lib.impl("masked_fill.Scalar", custom_implementation, "CUDA")
```

------

## **总结**

- **`CompositeExplicitAutograd` 的定位**：用于实现高层操作的复合逻辑，支持 Autograd。
- **作用**：为未特化的设备或后端提供默认实现，降低开发成本。
- **与调度的关系**：当没有设备特化实现时，PyTorch 自动回退到 `CompositeExplicitAutograd`。
- **注意事项**：用户注册自定义实现时，需要谨慎处理与复合操作的交互，确保行为符合预期。