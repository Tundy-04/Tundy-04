本文记录我认为有意义的或者可能以后会忘掉的PyTorch小知识

# 内存共享

在pytorch中有很多操作提供两个版本的函数：共享内存或者复制一份

这里收集一些可能会引起冲突的数据改变操作。比如你改了某个数据，但它可能会连带着其他共享同一片内存的其他相关变量一起改变了而不自知。它是内存优化带来的弊端，这些错误往往出现在一些非常基本的函数中，但我们没有细究过通过这些函数改变某个值意味着什么。

对于张量的冲突检查，你可以使用.storage().data_ptr()直接比较两个张量的地址来确认。但是提前了解那些操作是共用数据的，在新建张量时根据用途决定是否使用复制的方法来创建，避免一些最后查不出来的可能bug会更好。
```
y.storage().data_ptr() == x.storage().data_ptr()
```

## 用Numpy导致的

NumPy 数组和 PyTorch 张量可以方便转换，但其中有冲突的隐患。 

NumPy → PyTorch

使用**torch.from_numpy**()时，生成的 PyTorch 张量和原始 NumPy 数组在 CPU 上共享相同的底层内存位置。这意味着修改一个对象会影响另一个。这种行为很高效，因为它避免了数据复制，但你需要注意这一点。

```python
# 创建一个 NumPy 数组
numpy_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
print(f"NumPy 数组:\n{numpy_array}")
print(f"NumPy 数组类型: {numpy_array.dtype}")

# 将 NumPy 数组转换为 PyTorch 张量
pytorch_tensor = torch.from_numpy(numpy_array)
print(f"\nPyTorch 张量:\n{pytorch_tensor}")
print(f"PyTorch 张量类型: {pytorch_tensor.dtype}")

# 修改 NumPy 数组
numpy_array[0, 0] = 99
print(f"\n修改后的 NumPy 数组:\n{numpy_array}")
print(f"修改 NumPy 数组后的 PyTorch 张量:\n{pytorch_tensor}")

# 修改 PyTorch 张量
pytorch_tensor[1, 1] = -1
print(f"\n修改后的 PyTorch 张量:\n{pytorch_tensor}")
print(f"修改 PyTorch 张量后的 NumPy 数组:\n{numpy_array}")
```

<details>
  <summary>点我看输出</summary>
  <pre><code>
NumPy 数组:
[[1. 2.]
 [3. 4.]]
NumPy 数组类型: float32

PyTorch 张量:
tensor([[1., 2.],
        [3., 4.]])
PyTorch 张量类型: torch.float32

修改后的 NumPy 数组:
[[99.  2.]
 [ 3.  4.]]
修改 NumPy 数组后的 PyTorch 张量:
tensor([[99.,  2.],
        [ 3.,  4.]])

修改后的 PyTorch 张量:
tensor([[99.,  2.],
        [ 3., -1.]])
修改 PyTorch 张量后的 NumPy 数组:
[[99.  2.]
 [ 3. -1.]]
  </code></pre>
</details>

PyTorch → Numpy

反之，你可以使用 .numpy() 方法将位于 CPU 上的 PyTorch 张量转换回 NumPy 数组。同样，生成的 NumPy 数组和原始 **CPU** 张量共享相同的底层内存。对一个的修改会影响另一个。

```python
# 在 CPU 上创建一个 PyTorch 张量
cpu_tensor = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
print(f"原始 PyTorch 张量 (CPU):\n{cpu_tensor}")

# 将张量转换为 NumPy 数组
numpy_array_converted = cpu_tensor.numpy()
print(f"\n转换后的 NumPy 数组:\n{numpy_array_converted}")
print(f"NumPy 数组类型: {numpy_array_converted.dtype}")

# 修改张量
cpu_tensor[0, 1] = 25.0
print(f"\n修改后的 PyTorch 张量:\n{cpu_tensor}")
print(f"修改张量后的 NumPy 数组:\n{numpy_array_converted}")

# 修改 NumPy 数组
numpy_array_converted[1, 0] = 35.0
print(f"\n修改后的 NumPy 数组:\n{numpy_array_converted}")
print(f"修改 NumPy 数组后的张量:\n{cpu_tensor}")
```
<details>
  <summary>点我看输出</summary>
  <pre><code>
原始 PyTorch 张量 (CPU):
tensor([[10., 20.],
        [30., 40.]])

转换后的 NumPy 数组:
[[10. 20.]
 [30. 40.]]
NumPy 数组类型: float32

修改后的 PyTorch 张量:
tensor([[10., 25.],
        [30., 40.]])
修改张量后的 NumPy 数组:
[[10. 25.]
 [30. 40.]]

修改后的 NumPy 数组:
[[10. 25.]
 [35. 40.]]
修改 NumPy 数组后的张量:
tensor([[10., 25.],
        [35., 40.]])
  </code></pre>
</details>

.numpy() 方法仅适用于存储在 CPU 上的张量。如果你的张量在 GPU 上，你必须先使用 .cpu() 方法将其移到 CPU，然后才能将其转换为 NumPy 数组。直接在 GPU 张量上调用 .numpy() 会导致错误。

轻松地在 NumPy 数组和 PyTorch 张量之间转换的能力非常实用。你可能使用熟悉的 NumPy 函数或其他操作 NumPy 数组的库来执行初始数据加载和预处理。然后，当需要构建或训练深度学习模型时，你可以将数据转换为 PyTorch 张量，以借助于 GPU 加速和自动微分。同样，模型输出（即张量）可以转换回 NumPy 数组，以便使用 Matplotlib 或 Seaborn 等库进行分析或可视化。

## 原地操作函数

原地操作是指直接修改变量本身，而不创建新的对象。这种操作在深度学习中非常常见，尤其是在处理大规模数据时，可以显著提高效率并节省内存。在 PyTorch 中，原地操作的函数通常以 _ 结尾，例如 add_()、mul_() 等。以下是一个简单的示例：

```python
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
x.add_(y) # 原地加法操作
print(x) # 输出: tensor([5, 7, 9])
x.add_(y).mul_(2) # 原地加法和乘法
print(x) # 输出: tensor([18, 24, 30])
```

原地操作具有以下几个显著优势：

- 减少内存开销：由于不需要额外的内存空间存储结果，原地操作非常适合处理大规模数据集。

- 提高代码效率：避免了创建新对象的开销，从而加快了程序的执行速度。

- 支持链式操作：可以连续对同一变量进行多次操作，使代码更加简洁。

尽管原地操作有诸多优点，但在使用时需要注意以下几点：

- 修改原变量：原地操作会直接修改变量的值，因此在使用时需确保不会影响其他依赖该变量的操作。

- 梯度计算问题：在自动求导中，原地操作可能会覆盖计算梯度所需的值，从而导致梯度计算错误。

- 不可逆修改：由于原地操作直接修改变量，某些情况下可能导致数据不可恢复

## 切片操作

切片的一个重要特性（与某些其他形式的索引不同）是，返回的张量通常与原始张量共享底层存储。修改切片会修改原始张量。

```python
# 创建一个二维张量 (例如，一个小矩阵)
x_2d = torch.tensor([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
print(f"原始二维张量:\n{x_2d}")

print(f"修改切片前的原始 x_2d:\n{x_2d}")

# 获取一个切片
sub_tensor = x_2d[0:2, 1:3]

# 修改切片
sub_tensor[0, 0] = 101

print(f"\n修改后的切片:\n{sub_tensor}")
print(f"\n修改切片后的原始 x_2d:\n{x_2d}") # 注意变化！
```
<details>
  <summary>点我看输出</summary>
  <pre><code>
原始二维张量:
tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])
修改切片前的原始 x_2d:
tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])

修改后的切片:
tensor([[101,   3],
        [  5,   6]])

修改切片后的原始 x_2d:
tensor([[  1, 101,   3],
        [  4,   5,   6],
        [  7,   8,   9]])
  </code></pre>
</details>
如果您需要一个不共享内存的副本，请在切片上使用 .clone(): sub_tensor_copy = x_2d[0:2, 1:3].clone()

## View重塑张量

view() 方法返回一个新的张量，该张量与原始张量共享相同的底层数据，但具有不同的形状。它非常高效，因为它避免了数据复制。然而，view() 要求张量在内存中是连续的。一般用的比较多的是.reshape

连续张量是指其元素在内存中按维度顺序连续存储，没有间隙的张量。大多数新创建的张量是连续的，但某些操作（如切片或使用 t() 进行转置）会产生非连续张量。

你可以在 view() 调用中对一个维度使用 -1，PyTorch 将根据总元素数量和其它维度的尺寸自动推断出该维度的正确尺寸。
```python
# 创建一个连续张量
x = torch.arange(12) # tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
print(f"原始张量: {x}")
print(f"原始形状: {x.shape}")
print(f"是否连续? {x.is_contiguous()}")

# 使用 view() 重塑
y = x.view(3, 4)
print("\nview(3, 4) 后的张量:")
print(y)
print(f"新形状: {y.shape}")
print(f"与 x 共享存储吗? {y.storage().data_ptr() == x.storage().data_ptr()}") # 检查它们是否共享内存
print(f"y 是否连续? {y.is_contiguous()}")

# 尝试另一个视图
z = y.view(2, 6)
print("\nview(2, 6) 后的张量:")
print(z)
print(f"新形状: {z.shape}")
print(f"与 x 共享存储吗? {z.storage().data_ptr() == x.storage().data_ptr()}")
print(f"z 是否连续? {z.is_contiguous()}")

# 使用 -1 进行推断
w = x.view(2, 2, -1) # 推断出最后一个维度为 3 (12 / (2*2) = 3)
print("\nview(2, 2, -1) 后的张量:")
print(w)
print(f"新形状: {w.shape}")
```

<details>
  <summary>点我看输出</summary>
  <pre><code>
原始张量: tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
原始形状: torch.Size([12])
是否连续? True

view(3, 4) 后的张量:
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
新形状: torch.Size([3, 4])
与 x 共享存储吗? True
y 是否连续? True

view(2, 6) 后的张量:
tensor([[ 0,  1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10, 11]])
新形状: torch.Size([2, 6])
与 x 共享存储吗? True
z 是否连续? True

view(2, 2, -1) 后的张量:
tensor([[[ 0,  1,  2],
         [ 3,  4,  5]],

        [[ 6,  7,  8],
         [ 9, 10, 11]]])
新形状: torch.Size([2, 2, 3])
  </code></pre>
</details>

使用 permute() 调整维度顺序：

和 view() 一样，permute() 返回一个与原始张量共享底层数据的张量。它不复制数据。然而，生成的张量通常不是连续的。如果你在调换维度后需要一个连续张量（例如，为了后续使用 view()），你可以链式调用 .contiguous() 方法

# 张量操作

这里记录一些张量操作

## 索引和切片

### 切片

切片允许您沿张量维度选择一系列元素。语法是 start:stop:step，其中 start 是包含的，stop 是**不包含**的，而 step 定义了间隔。省略 start 默认为0，省略 stop 默认为维度的末尾，省略 step 默认为1。

当step设置为-1时，可以简单地翻转张量
```python
# 创建一个一维张量
y_1d = torch.arange(10) # Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(f"原始一维张量: {y_1d}")

# 选择每隔一个的元素
slice4 = y_1d[::2]
print(f"切片 y_1d[::2]: {slice4}")

# 选择从索引1到7的元素，步长为2
slice5 = y_1d[1:8:2]
print(f"切片 y_1d[1:8:2]: {slice5}")

# 反转张量
slice6 = y_1d[::-1]
print(f"切片 y_1d[::-1]: {slice6}")
```
切片的一个重要特性（与某些其他形式的索引不同）是，返回的张量通常与原始张量共享底层存储。修改切片会修改原始张量。在上面有提到。

### 数组索引

#### 布尔索引 (遮罩)：

您可以使用布尔张量来索引另一个张量。布尔张量的形状必须能够广播到被索引张量的形状（通常，它们的形状完全相同）。只有布尔张量中对应 True 值的元素（即“遮罩”）才会被选中。这对于根据条件筛选数据非常有用。

布尔索引通常返回一个包含所有选定元素的**一维张量**。与切片不同，它不保留原始形状。此外，布尔索引通常会创建一个副本，而不是一个视图。

```python
# 创建一个张量
data = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(f"原始数据张量:\n{data}")

# 创建一个布尔遮罩 (例如，选择大于3的元素)
mask = data > 3
print(f"\n布尔遮罩 (data > 3):\n{mask}")

# 应用遮罩
selected_elements = data[mask]
print(f"\n通过遮罩选择的元素:\n{selected_elements}")
print(f"所选元素的形状: {selected_elements.shape}")

# 根据条件修改元素
data[data <= 3] = 0
print(f"\n将小于等于3的元素设置为零后的数据:\n{data}")

# 选择第一列大于2的行
row_mask = data[:, 0] > 2
print(f"\n行遮罩 (data[:, 0] > 2): {row_mask}")

selected_rows = data[row_mask, :] # 使用 ':' 选择所选行中的所有列
# Or simply: data[row_mask] - PyTorch 通常会推断出完整的行选择
print(f"\n第一列大于2的行:\n{selected_rows}")
```

<details>
  <summary>点我看输出</summary>
  <pre><code>
原始数据张量:
tensor([[1, 2],
        [3, 4],
        [5, 6]])

布尔遮罩 (data > 3):
tensor([[False, False],
        [False,  True],
        [ True,  True]])

通过遮罩选择的元素:
tensor([4, 5, 6])
所选元素的形状: torch.Size([3])

将小于等于3的元素设置为零后的数据:
tensor([[0, 0],
        [0, 4],
        [5, 6]])

行遮罩 (data[:, 0] > 2): tensor([False, False,  True])

第一列大于2的行:
tensor([[5, 6]])
  </code></pre>
</details>

#### 整数数组索引

除了单个整数和切片，您还可以使用列表或一维整数张量沿维度进行索引。这使得您可以按任意顺序选择元素，或多次选择相同的元素。

与布尔索引类似，整数数组索引通常返回一个新的张量（一个副本），而不是原始张量存储的视图。输出的**形状取决于索引方法**。当选择完整的行或列时，其他维度会被保留。当为多个维度提供索引数组（例如 y[row_idx, col_idx]）时，结果通常是一个对应于所选元素的一维张量。

```python
x = torch.arange(10, 20) # Tensor([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
print(f"原始一维张量: {x}")

indices = torch.tensor([0, 4, 2, 2]) # 注意索引2的重复
selected = x[indices]
print(f"\n使用索引 {indices} 选择的元素: {selected}")

# 对于二维张量
y = torch.arange(12).reshape(3, 4)
# [[ 0,  1,  2,  3],
#  [ 4,  5,  6,  7],
#  [ 8,  9, 10, 11]]
print(f"\n原始二维张量:\n{y}")

# 选择特定行
row_indices = torch.tensor([0, 2])
selected_rows = y[row_indices] # 选择第0行和第2行
print(f"\n使用索引 {row_indices} 选择的行:\n{selected_rows}")

# 选择特定列
col_indices = torch.tensor([1, 3])
selected_cols = y[:, col_indices] # 从所有行中选择第1列和第3列
print(f"\n使用索引 {col_indices} 选择的列:\n{selected_cols}")

# 使用索引对选择特定元素
row_idx = torch.tensor([0, 1, 2])
col_idx = torch.tensor([1, 3, 0])
selected_elements = y[row_idx, col_idx] # 选择 (0,1), (1,3), (2,0) -> [1, 7, 8]
print(f"\n使用 (row_idx, col_idx) 选择的特定元素:\n{selected_elements}")
```

<details>
  <summary>点我看输出</summary>
  <pre><code>
原始一维张量: tensor([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

使用索引 tensor([0, 4, 2, 2]) 选择的元素: tensor([10, 14, 12, 12])

原始二维张量:
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])

使用索引 tensor([0, 2]) 选择的行:
tensor([[ 0,  1,  2,  3],
        [ 8,  9, 10, 11]])

使用索引 tensor([1, 3]) 选择的列:
tensor([[ 1,  3],
        [ 5,  7],
        [ 9, 11]])

使用 (row_idx, col_idx) 选择的特定元素:
tensor([1, 7, 8])
  </code></pre>
</details>