[PennyLane](https://pennylane.ai/qml) 是一个将量子计算与机器学习融合的开源库，其核心设计理念是通过可微分量子电路无缝集成经典与量子计算，其对量子设备和经典设备兼容性较好，同时兼容pytorch。下面我们详细解析如何使用 QNode搭建量子线路、常用的量子门函数、可视化方法，以及和pytorch的model中作为一个层的接轨和训练。

# 1.量子线路搭建

这里介绍如何搭建一个量子线路，包括静态的和动态的（PQC），可以广泛用于量子计算相关代码实现，包括量子算法，量子机器学习等。

```python
import pennylane as qml
```
一般库的缩写是名字或用途的缩写，这里缩写习惯用qml（量子机器学习）

## 1.1 基本结构

⚛️ QNode：量子计算的核心单元

QNode是 PennyLane 的核心抽象，它将量子函数（包含量子操作的 Python 函数）转换为可在量子设备（模拟器或真实硬件）上执行的对象，并确保其可微性，以支持梯度计算和优化。

构建一个基本的 QNode包含三个关键步骤：

- **定义量子设备**：指定运行量子线路的后端。
```python
import pennylane as qml
# 创建一个使用 2 个量子比特的默认量子模拟器
dev = qml.device("default.qubit", wires=2)
```
这里的 wires参数定义了可用的量子比特数量(线路数目)
- **使用 @qml.qnode装饰器**：将经典的 Python 函数转换为量子节点。
```python
@qml.qnode(dev)  # 将 quantum_circuit 函数绑定到设备 dev
def quantum_circuit(x):
    # 量子操作将在这里定义
    ...
    return ...  # 返回测量结果
```
- **在函数内定义量子线路**：在被装饰的函数中，通过添加量子门操作来构建线路，并最终返回一个测量值。
```python
@qml.qnode(dev)
def quantum_circuit(x):
    qml.Hadamard(wires=0)       # 对第0个量子比特应用H门，创建叠加态
    qml.RY(x, wires=0)          # 使用参数 x 进行Y轴旋转
    qml.CNOT(wires=[0, 1])      # 在量子比特0和1之间应用CNOT门，生成纠缠
    return qml.expval(qml.PauliZ(0))  # 返回第一个量子比特的Pauli-Z期望值
```
执行此函数（如 result = quantum_circuit(0.5)）会在指定的设备上运行线路并返回测量结果

<img width="1387" height="1820" alt="Image" src="https://github.com/user-attachments/assets/4a671b06-9900-4c72-923c-1105a4cbc027" />

## 1.2 测量线路

| 测量类型(QNode返回) | 函数                     | 输出含义                                               | 经典类比(近似理解)                               |
| ------------------- | ------------------------ | ------------------------------------------------------ | ------------------------------------------------ |
| **期望值**          | `qml.expval(Observable)` | 对某个可观测量（如自旋方向）的平均值。                 | 预测一个随机变量的**平均值**。                   |
| **概率**            | `qml.probs(wires)`       | 测量得到每个计算基态的概率。                           | 一个概率分布，如 `[0.7, 0.3]`。                  |
| **样本**            | `qml.sample(Observable)` | 对可观测量进行多次重复测量，得到具体结果。             | 掷骰子N次，记录每次的点数。                     |
| **量子态**          | `qml.state()`            | 直接获取模拟的量子态向量（仅仿真器可用）。             | 获取一个随机过程的完整概率幅。                   |

🔬需要说明的是：直接获取模拟的量子态向量，能不用就尽量不用，毕竟量子计算研究的目的最后是应用在真实的量子硬件上的。量子态会在测量后坍缩，这是不可逆的，如果想知道某个具体的量子态，测量后它就失去意义了。而根据不可克隆定理，你没法通过简单的开销（需要额外的辅助qubit）就造出一个一样的量子态来。仿真中你当然可以直接获取某个qubit的密度矩阵来复制它和测量它，但在真实硬件中没法这么做，属于是“作弊”行为。

## 1.3 量子门函数

量子门是操控量子态的基本单元。PennyLane 提供了丰富的门函数，以下是一些核心类别：

| 门类型   | 函数示例                         | 功能描述                                                     |
| -------- | -------------------------------- | ------------------------------------------------------------ |
| Pauli门  | `qml.PauliX(wires=i)`            | 比特翻转（绕X轴旋转π）                                       |
|          | `qml.PauliY(wires=i)`            | 绕Y轴旋转π                                                  |
|          | `qml.PauliZ(wires=i)`            | 相位翻转（绕Z轴旋转π）                                       |
| 旋转门   | `qml.RX(phi, wires=i)`           | 绕X轴旋转角度φ                                              |
|          | `qml.RY(phi, wires=i)`           | 绕Y轴旋转角度φ                                              |
|          | `qml.RZ(phi, wires=i)`           | 绕Z轴旋转角度φ                                              |
| 通用门   | `qml.Hadamard(wires=i)`          | 创建叠加态: \|0⟩ → (\|0⟩+\|1⟩)/√2                           |
|          | `qml.CNOT(wires=[c, t])`         | 受控非门：控制比特c为1时翻转目标比特t                       |
| 特殊门   | `qml.Rot(a, b, c, wires=i)`      | 依次进行RZ(b)、RY(a)、RZ(c)旋转，用于实现任意单比特酉变换 。 |

多量子比特门（如 CNOT）是产生量子纠缠的关键。所有这些门操作都按其在函数中出现的顺序依次执行

## 1.4 线路可视化

📐 量子线路可视化

清晰地查看线路结构对于理解和调试至关重要。PennyLane 提供了便捷的可视化工具。

- 基础文本可视化：使用 qml.draw()函数可以生成线路的文本图。

```python
# 接上面的代码
drawn_circuit = qml.draw(quantum_circuit)(0.5)
print(drawn_circuit)
```
- 高级绘图功能：对于更复杂的线路，可以使用 qml.draw_mpl函数生成更精美的 matplotlib 图像。

```python
fig, ax = qml.draw_mpl(quantum_circuit, decimals=2)(0.5)
plt.show()
```
## 1.5 示例

以下是一段量子线路实现和可视化的完整示例：

```python
# 1. 定义量子设备（使用默认的模拟器）
dev = qml.device("default.qubit", wires=2)

# 2. 创建量子节点，并指定使用PyTorch接口
@qml.qnode(dev, interface="torch")
def circuit(params):
    qml.RX(params[0], wires=0)  # 用参数params[0]进行X旋转
    qml.RY(params[1], wires=1)  # 用参数params[1]进行Y旋转
    qml.CNOT(wires=[0, 1])      # 添加一个CNOT门
    return qml.expval(qml.PauliZ(0))  # 返回量子比特0上Pauli Z算符的期望值

# 3. 可视化
print("\n=== 量子线路图形可视化 ===")
try:
    # 使用qml.draw_mpl()生成图形化电路图[1,6](@ref)
    # 使用Windows系统自带的微软雅黑（适用于Windows用户）
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或 ['SimHei'] 对应黑体
    fig, ax = qml.draw_mpl(circuit)(params)
    plt.title("参数化量子电路可视化")
    plt.tight_layout()
    plt.show()
    print("图形可视化已显示，关闭窗口继续执行...")
except Exception as e:
    print(f"图形可视化失败: {e}")
    print("请确保已安装matplotlib: pip install matplotlib")
```
输出：

<img width="519" height="342" alt="Image" src="https://github.com/user-attachments/assets/e59ec9e3-db03-48fa-a5b9-778da823514d" />

# 2.搭建量子层

搭建量子层（随机静态/PQC）的逻辑流程如下：

<img width="2119" height="1964" alt="Image" src="https://github.com/user-attachments/assets/93acf066-ada4-42a5-9deb-6f8d82178831" />

## 2.1 区别与选择

理解可训练的参数化量子电路（PQC）和静态量子线路的区别，是掌握量子机器学习的关键。下面这个表格能帮你快速抓住它们的核心不同。

| 特性维度         | 参数化量子电路(PQC)                                 | 静态量子线路                                       |
| ---------------- | --------------------------------------------------- | -------------------------------------------------- |
| 核心参数         | 包含可训练的参数（如旋转门角度θ）                   | 由固定的量子门序列组成，无训练参数                 |
| 设计目标         | 寻求最优解，通过训练调整参数以最小化损失函数        | 执行确定运算，实现设计好的特定量子逻辑             |
| 与经典系统交互   | 深度耦合，构成量子-经典混合计算闭环的一部分          | 弱交互，通常一次性输入经典数据并输出结果           |
| 测量输出         | 测量结果（如期望值）用于计算损失函数，并指导参数更新 | 测量结果即为计算的最终答案                         |
| 典型应用         | 量子机器学习、变分量子算法（VQE, QAOA）             | 量子傅里叶变换、Grover搜索算法等                   |

🧠 可训练PQC的运作方式

PQC的强大之处在于它和经典机器学习流程的无缝融合。

电路构建：首先，你需要设计一个线路架构（Ansatz）。这个架构中包含了一些由参数控制的量子门，最常见的是单量子比特旋转门（如 RX(θ), RY(θ), RZ(θ)），其中的角度 θ就是可训练的参数。

混合计算闭环：PQC的运作是一个典型的“量子-经典混合”过程：

- 经典侧：初始化参数，将参数 θ和输入数据（通过特定编码方式注入量子态）传给量子设备。
- 量子侧：在量子设备上运行PQC，并进行测量，得到期望值等结果。
- 经典侧：根据测量结果计算损失函数，然后利用经典优化器（如梯度下降）根据梯度信息更新参数 θ。这个循环会持续进行，直到模型收敛。

在PennyLane中，@qml.qnode装饰器封装了量子计算部分，而自动微分功能使得计算这些梯度变得非常简单。

⚙️ 静态线路的确定性逻辑

静态线路的设计和运行逻辑与PQC有本质区别。

固定流程：静态线路的门序列和所有参数在构建时就已经完全确定。每次运行同一线路，只要输入相同，得到的输出在概率分布上就是一致的。

核心在于设计：对于静态线路，工作的重点在于设计出正确的门序列来实现特定算法，例如在量子傅里叶变换中精确安排受控门的位置和相位旋转门的角度。

💡 如何选择？

选择PQC：当你面对的问题没有已知的高效量子算法，或者需要在含噪声的中等规模量子（NISQ）设备上解决优化、机器学习任务时，PQC和变分量子算法是首选。

选择静态线路：当你需要执行已被严格证明具有量子优势的算法（如Shor算法、Grover算法）时，静态线路是实现这些算法的标准方式。

## 2.2 搭建静态与动态量子层

💻 代码实现详解

下面我们看看在PyTorch和PennyLane的混合模型中，这两种线路如何具体实现。

### 公共设置
首先，我们需要导入必要的库并定义一些公共参数。

```python
import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

# 定义量子设备
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

# 定义用于数据编码的量子节点（公共部分）
@qml.qnode(dev, interface="torch")
def data_encoder(inputs):
    # 将经典数据编码到量子态，例如通过旋转门
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)  # 使用输入数据的每个元素作为旋转角度
    return qml.state()
```

### 静态线路实现
静态线路的参数是固定的，不参与训练。

```python
# --- 静态线路部分 ---
# 定义一个固定的、不训练的参数
fixed_angle = torch.tensor([np.pi / 4])  # 例如，固定为45度

@qml.qnode(dev, interface="torch")
def static_quantum_conv(inputs):
    # 1. 数据编码（与PQC共享）
    data_encoder(inputs)
    
    # 2. 应用静态的量子卷积操作
    # 例如，使用固定的角度和固定的纠缠结构
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])
    qml.RZ(fixed_angle, wires=n_qubits-1)  # 使用固定参数
    
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# 定义包含静态量子层的PyTorch模型
class QCNN_Static(nn.Module):
    def __init__(self):
        super().__init__()
        # 注意：静态线路的参数（如fixed_angle）不注册为nn.Parameter
        self.classical_fc = nn.Linear(n_qubits, 1)  # 后续经典层
        
    def forward(self, x):
        # 量子部分：静态线路，每次计算行为一致
        quantum_out = torch.tensor([static_quantum_conv(x_i) for x_i in x])
        # 经典部分
        output = self.classical_fc(quantum_out)
        return output
```

### 参数化量子线路（PQC）实现

PQC的核心在于其参数是可训练的。
```python
# --- PQC部分 ---
# 定义可训练参数的维度：假设有2层，每层为每个量子比特准备一个参数
depth = 2
quantum_weights_shape = (depth, n_qubits)

@qml.qnode(dev, interface="torch")
def pqc_quantum_conv(inputs, weights):  # 注意：线路函数接收权重参数
    # 1. 数据编码（与静态线路共享）
    data_encoder(inputs)
    
    # 2. 应用参数化的量子卷积操作（变分层）
    # 使用可训练的权重！这是与静态线路最根本的区别。
    for layer in range(depth):
        for i in range(n_qubits):
            qml.RY(weights[layer][i], wires=i)
        # 添加纠缠以增加复杂性
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i+1])
    
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# 定义包含PQC的PyTorch模型
class QCNN_PQC(nn.Module):
    def __init__(self):
        super().__init__()
        # 关键区别：将量子线路的权重注册为PyTorch模型参数
        self.quantum_weights = nn.Parameter(torch.randn(quantum_weights_shape))
        self.classical_fc = nn.Linear(n_qubits, 1)
        
    def forward(self, x):
        # 量子部分：PQC，将可训练参数quantum_weights传入量子节点
        # 每次前向传播都使用当前优化器更新后的权重值
        quantum_out = torch.stack([pqc_quantum_conv(x_i, self.quantum_weights) for x_i in x])
        # 经典部分
        output = self.classical_fc(quantum_out)
        return output
```

> 最直观的区别:静态线路的参数是普通Tensor，而PQC的参数是nn.Parameter，这会告诉PyTorch这些值需要在训练中被优化。