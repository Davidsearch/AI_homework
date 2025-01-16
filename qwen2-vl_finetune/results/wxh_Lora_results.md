ResNet18、VGG11

- Lora:

15657MiB /  24576MiB

- AdaLoRA_epoch=1:

Training Epoch 1: 100%|████████████████████████████████████████████████████████████████████████████████████| 2929/2929 [4:45:25<00:00,  5.85s/it]
Epoch 1 Loss: 3.0146

22875MiB /  24576MiB 

- LoRa+deepspeed stage 2:
  18767MiB /  24576MiB

- Lora+deepspeed stage 3:

  ![image-20250109112345511](C:\Users\86139\AppData\Roaming\Typora\typora-user-images\image-20250109112345511.png)

```文字
1. ZeRO 阶段 3 的额外开销
ZeRO 阶段 3 引入了更多的显存优化机制，例如将模型参数、梯度和优化器状态分片存储，并支持将部分数据卸载到 CPU。然而，这些机制也会带来额外的开销：

通信开销：

ZeRO 阶段 3 需要在多个 GPU 之间频繁通信，以同步分片的参数和梯度。这种通信可能会占用额外的显存和带宽。

如果通信效率不高，可能会导致显存占用增加。

分片管理开销：

ZeRO 阶段 3 需要对模型参数进行分片管理，这可能会引入额外的显存开销。

如果分片数量过多，或者分片大小不合适，可能会导致显存占用增加。

Offloading 开销：

如果启用了 Offloading（将部分数据卸载到 CPU），DeepSpeed 需要在 GPU 和 CPU 之间频繁传输数据。这种数据传输可能会占用额外的显存和带宽。

2. 模型和数据的特殊性
模型结构：

如果你的模型结构特殊（例如 Qwen2-VL 是一个多模态模型），ZeRO 阶段 3 的分片机制可能无法有效减少显存占用。

某些模型参数（如视觉模块的参数）可能不适合分片存储，导致显存占用增加。

数据大小：

如果你的数据（如图像或文本）较大，ZeRO 阶段 3 的分片机制可能无法有效减少显存占用。

数据加载和预处理可能会占用大量显存，尤其是在多模态任务中。

3. DeepSpeed 配置问题
Offloading 未正确启用：

如果你配置了 Offloading，但 DeepSpeed 未正确启用，可能会导致显存占用增加。

检查 DeepSpeed 日志，确认是否打印了 Offloading optimizer to CPU 或 Offloading parameters to CPU。

分片配置不合适：

ZeRO 阶段 3 的分片配置（如 stage3_max_live_parameters 和 stage3_max_reuse_distance）可能不适合你的模型和数据。

如果分片配置不合适，可能会导致显存占用增加。

4. 硬件限制
GPU 型号：

某些 GPU 型号（如 Tesla P40）可能不完全支持 ZeRO 阶段 3 的优化机制，导致显存占用增加。

检查你的 GPU 是否支持 DeepSpeed 的所有功能。

CPU 和 GPU 之间的带宽：

如果 CPU 和 GPU 之间的带宽不足，Offloading 可能会导致显存占用增加。

检查你的硬件配置，确保 CPU 和 GPU 之间的带宽足够。
```

zero stage = 3

[2025-01-09 10:30:34,379] [INFO] [logging.py:128:log_dist] [Rank 0] Using DeepSpeed Optimizer param name adam as basic optimizer
[2025-01-09 10:30:34,379] [INFO] [logging.py:128:log_dist] [Rank 0] Removing param_group that has no 'params' in the basic Optimizer
[2025-01-09 10:30:34,723] [INFO] [logging.py:128:log_dist] [Rank 0] DeepSpeed Basic Optimizer = DeepSpeedCPUAdam
[2025-01-09 10:30:34,723] [INFO] [utils.py:59:is_zero_supported_optimizer] Checking ZeRO support for optimizer=DeepSpeedCPUAdam type=<class 'deepspeed.ops.adam.cpu_adam.DeepSpeedCPUAdam'>
[2025-01-09 10:30:34,724] [INFO] [logging.py:128:log_dist] [Rank 0] Creating fp16 ZeRO stage 3 optimizer, MiCS is enabled False, Hierarchical params gather False
[2025-01-09 10:30:34,724] [INFO] [logging.py:128:log_dist] [Rank 0] Creating torch.float32 ZeRO stage 3 optimizer
[2025-01-09 10:30:34,904] [INFO] [utils.py:781:see_memory_usage] Stage 3 initialize beginning
[2025-01-09 10:30:34,904] [INFO] [utils.py:782:see_memory_usage] MA 4.17 GB         Max_MA 4.17 GB         CA 4.47 GB         Max_CA 4 GB 
[2025-01-09 10:30:34,905] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 3.68 GB, percent = 5.9%
[2025-01-09 10:30:34,922] [INFO] [stage3.py:168:__init__] Reduce bucket size 10000000
[2025-01-09 10:30:34,922] [INFO] [stage3.py:169:__init__] Prefetch bucket size 10000000
[2025-01-09 10:30:35,076] [INFO] [utils.py:781:see_memory_usage] DeepSpeedZeRoOffload initialize [begin]
[2025-01-09 10:30:35,076] [INFO] [utils.py:782:see_memory_usage] MA 4.17 GB         Max_MA 4.17 GB         CA 4.47 GB         Max_CA 4 GB 
[2025-01-09 10:30:35,077] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 3.68 GB, percent = 5.9%
[2025-01-09 10:30:35,102] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 1
Parameter Offload: Total persistent parameters: 5506036 in 1101 params
[2025-01-09 10:30:38,383] [INFO] [utils.py:781:see_memory_usage] DeepSpeedZeRoOffload initialize [end]
[2025-01-09 10:30:38,384] [INFO] [utils.py:782:see_memory_usage] MA 0.0 GB         Max_MA 4.17 GB         CA 4.47 GB         Max_CA 4 GB 
[2025-01-09 10:30:38,384] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 7.8 GB, percent = 12.4%
[2025-01-09 10:30:38,557] [INFO] [utils.py:781:see_memory_usage] Before creating fp16 partitions
[2025-01-09 10:30:38,558] [INFO] [utils.py:782:see_memory_usage] MA 0.0 GB         Max_MA 0.0 GB         CA 4.47 GB         Max_CA 4 GB 
[2025-01-09 10:30:38,558] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 7.8 GB, percent = 12.4%
[2025-01-09 10:30:38,772] [INFO] [utils.py:781:see_memory_usage] After creating fp16 partitions: 1
[2025-01-09 10:30:38,773] [INFO] [utils.py:782:see_memory_usage] MA 0.0 GB         Max_MA 0.0 GB         CA 4.47 GB         Max_CA 4 GB 
[2025-01-09 10:30:38,773] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 7.83 GB, percent = 12.5%
[2025-01-09 10:30:38,951] [INFO] [utils.py:781:see_memory_usage] Before creating fp32 partitions
[2025-01-09 10:30:38,952] [INFO] [utils.py:782:see_memory_usage] MA 0.0 GB         Max_MA 0.0 GB         CA 4.47 GB         Max_CA 4 GB 
[2025-01-09 10:30:38,952] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 7.83 GB, percent = 12.5%
[2025-01-09 10:30:39,138] [INFO] [utils.py:781:see_memory_usage] After creating fp32 partitions
[2025-01-09 10:30:39,138] [INFO] [utils.py:782:see_memory_usage] MA 0.0 GB         Max_MA 0.0 GB         CA 4.47 GB         Max_CA 4 GB 
[2025-01-09 10:30:39,139] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 7.91 GB, percent = 12.6%
[2025-01-09 10:30:39,314] [INFO] [utils.py:781:see_memory_usage] Before initializing optimizer states
[2025-01-09 10:30:39,314] [INFO] [utils.py:782:see_memory_usage] MA 0.0 GB         Max_MA 0.0 GB         CA 4.47 GB         Max_CA 4 GB 
[2025-01-09 10:30:39,315] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 7.91 GB, percent = 12.6%
[2025-01-09 10:30:39,505] [INFO] [utils.py:781:see_memory_usage] After initializing optimizer states
[2025-01-09 10:30:39,506] [INFO] [utils.py:782:see_memory_usage] MA 0.0 GB         Max_MA 0.0 GB         CA 4.47 GB         Max_CA 4 GB 
[2025-01-09 10:30:39,506] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 8.03 GB, percent = 12.8%
[2025-01-09 10:30:39,507] [INFO] [stage3.py:528:_setup_for_real_optimizer] optimizer state initialized
[2025-01-09 10:30:39,920] [INFO] [utils.py:781:see_memory_usage] After initializing ZeRO optimizer
[2025-01-09 10:30:39,921] [INFO] [utils.py:782:see_memory_usage] MA 0.02 GB         Max_MA 0.02 GB         CA 4.47 GB         Max_CA 4 GB 
[2025-01-09 10:30:39,921] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 8.05 GB, percent = 12.9%
[2025-01-09 10:30:39,921] [INFO] [logging.py:128:log_dist] [Rank 0] DeepSpeed Final Optimizer = DeepSpeedZeroOptimizer_Stage3
[2025-01-09 10:30:39,921] [INFO] [logging.py:128:log_dist] [Rank 0] DeepSpeed using configured LR scheduler = None
[2025-01-09 10:30:39,921] [INFO] [logging.py:128:log_dist] [Rank 0] DeepSpeed LR Scheduler = None
[2025-01-09 10:30:39,921] [INFO] [logging.py:128:log_dist] [Rank 0] step=0, skipped=0, lr=[5e-05], mom=[[0.9, 0.999]]

## 解释

这段日志是 DeepSpeed 在初始化过程中输出的详细信息，展示了 DeepSpeed 如何配置和优化训练过程。以下是对日志的逐条解释：

---

### 1. **优化器配置**
- **日志**：
  ```
  [2025-01-09 10:30:34,379] [INFO] [logging.py:128:log_dist] [Rank 0] Using DeepSpeed Optimizer param name adam as basic optimizer
  [2025-01-09 10:30:34,379] [INFO] [logging.py:128:log_dist] [Rank 0] Removing param_group that has no 'params' in the basic Optimizer
  [2025-01-09 10:30:34,723] [INFO] [logging.py:128:log_dist] [Rank 0] DeepSpeed Basic Optimizer = DeepSpeedCPUAdam
  ```
- **解释**：
  - DeepSpeed 使用了 `adam` 作为基础优化器。
  - 移除了没有参数的优化器组（`param_group`）。
  - 最终选择了 `DeepSpeedCPUAdam` 作为基础优化器。

---

### 2. **ZeRO 优化配置**
- **日志**：
  ```
  [2025-01-09 10:30:34,724] [INFO] [logging.py:128:log_dist] [Rank 0] Creating fp16 ZeRO stage 3 optimizer, MiCS is enabled False, Hierarchical params gather False
  [2025-01-09 10:30:34,724] [INFO] [logging.py:128:log_dist] [Rank 0] Creating torch.float32 ZeRO stage 3 optimizer
  ```
- **解释**：
  - DeepSpeed 启用了 ZeRO 优化，并选择了 **Stage 3**。
  - 使用了混合精度训练（`fp16`）和单精度（`torch.float32`）优化器。

---

### 3. **内存使用情况**
- **日志**：
  ```
  [2025-01-09 10:30:34,904] [INFO] [utils.py:781:see_memory_usage] Stage 3 initialize beginning
  [2025-01-09 10:30:34,904] [INFO] [utils.py:782:see_memory_usage] MA 4.17 GB         Max_MA 4.17 GB         CA 4.47 GB         Max_CA 4 GB 
  [2025-01-09 10:30:34,905] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 3.68 GB, percent = 5.9%
  ```
- **解释**：
  - 初始化 ZeRO Stage 3 优化器时，记录了当前的内存使用情况。
  - `MA` 表示当前分配的 GPU 内存，`Max_MA` 表示最大分配的 GPU 内存。
  - `CA` 表示当前使用的 GPU 内存，`Max_CA` 表示最大使用的 GPU 内存。
  - CPU 虚拟内存使用量为 3.68 GB，占用率为 5.9%。

---

### 4. **ZeRO Stage 3 初始化**
- **日志**：
  ```
  [2025-01-09 10:30:34,922] [INFO] [stage3.py:168:__init__] Reduce bucket size 10000000
  [2025-01-09 10:30:34,922] [INFO] [stage3.py:169:__init__] Prefetch bucket size 10000000
  ```
- **解释**：
  - 设置了 ZeRO Stage 3 的 **Reduce 桶大小** 和 **Prefetch 桶大小** 为 10000000。

---

### 5. **参数卸载（Parameter Offload）**
- **日志**：
  ```
  [2025-01-09 10:30:35,102] [INFO] [config.py:733:__init__] Config mesh_device None world_size = 1
  Parameter Offload: Total persistent parameters: 5506036 in 1101 params
  ```
- **解释**：
  - 启用了参数卸载（Parameter Offload），将部分模型参数卸载到 CPU 上。
  - 总共有 5506036 个持久化参数，分布在 1101 个参数组中。

---

### 6. **内存使用情况（ZeRO Offload 初始化）**
- **日志**：
  ```
  [2025-01-09 10:30:38,384] [INFO] [utils.py:781:see_memory_usage] DeepSpeedZeRoOffload initialize [end]
  [2025-01-09 10:30:38,384] [INFO] [utils.py:782:see_memory_usage] MA 0.0 GB         Max_MA 4.17 GB         CA 4.47 GB         Max_CA 4 GB 
  [2025-01-09 10:30:38,384] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 7.8 GB, percent = 12.4%
  ```
- **解释**：
  - ZeRO Offload 初始化完成后，GPU 内存使用量降为 0.0 GB，CPU 内存使用量增加到 7.8 GB。

---

### 7. **FP16 分区创建**
- **日志**：
  ```
  [2025-01-09 10:30:38,772] [INFO] [utils.py:781:see_memory_usage] After creating fp16 partitions: 1
  [2025-01-09 10:30:38,773] [INFO] [utils.py:782:see_memory_usage] MA 0.0 GB         Max_MA 0.0 GB         CA 4.47 GB         Max_CA 4 GB 
  [2025-01-09 10:30:38,773] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 7.83 GB, percent = 12.5%
  ```
- **解释**：
  - 创建了 FP16 分区，GPU 内存使用量保持为 0.0 GB，CPU 内存使用量略微增加。

---

### 8. **FP32 分区创建**
- **日志**：
  ```
  [2025-01-09 10:30:39,138] [INFO] [utils.py:781:see_memory_usage] After creating fp32 partitions
  [2025-01-09 10:30:39,138] [INFO] [utils.py:782:see_memory_usage] MA 0.0 GB         Max_MA 0.0 GB         CA 4.47 GB         Max_CA 4 GB 
  [2025-01-09 10:30:39,139] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 7.91 GB, percent = 12.6%
  ```
- **解释**：
  - 创建了 FP32 分区，GPU 内存使用量保持为 0.0 GB，CPU 内存使用量进一步增加。

---

### 9. **优化器状态初始化**
- **日志**：
  ```
  [2025-01-09 10:30:39,506] [INFO] [utils.py:782:see_memory_usage] MA 0.0 GB         Max_MA 0.0 GB         CA 4.47 GB         Max_CA 4 GB 
  [2025-01-09 10:30:39,506] [INFO] [utils.py:789:see_memory_usage] CPU Virtual Memory:  used = 8.03 GB, percent = 12.8%
  [2025-01-09 10:30:39,507] [INFO] [stage3.py:528:_setup_for_real_optimizer] optimizer state initialized
  ```
- **解释**：
  - 初始化了优化器状态，GPU 内存使用量保持为 0.0 GB，CPU 内存使用量增加到 8.03 GB。

---

### 10. **最终优化器配置**
- **日志**：
  ```
  [2025-01-09 10:30:39,921] [INFO] [logging.py:128:log_dist] [Rank 0] DeepSpeed Final Optimizer = DeepSpeedZeroOptimizer_Stage3
  [2025-01-09 10:30:39,921] [INFO] [logging.py:128:log_dist] [Rank 0] DeepSpeed using configured LR scheduler = None
  [2025-01-09 10:30:39,921] [INFO] [logging.py:128:log_dist] [Rank 0] DeepSpeed LR Scheduler = None
  [2025-01-09 10:30:39,921] [INFO] [logging.py:128:log_dist] [Rank 0] step=0, skipped=0, lr=[5e-05], mom=[[0.9, 0.999]]
  ```
- **解释**：
  - 最终使用的优化器是 `DeepSpeedZeroOptimizer_Stage3`。
  - 未配置学习率调度器（`LR scheduler = None`）。
  - 当前学习率为 `5e-05`，动量（momentum）为 `[0.9, 0.999]`。

---

### 总结
这段日志展示了 DeepSpeed 在初始化过程中如何配置优化器、启用 ZeRO Stage 3 和参数卸载，并记录了内存使用情况。关键点包括：
1. 使用了 `DeepSpeedCPUAdam` 作为基础优化器。
2. 启用了 ZeRO Stage 3 和参数卸载，显著减少了 GPU 内存占用。
3. 未配置学习率调度器。
4. 记录了详细的内存使用情况，帮助用户了解资源分配。

如果有其他问题，欢迎继续提问！

## Lora (无deepspeed):

![image-20250109193315059](C:\Users\86139\AppData\Roaming\Typora\typora-user-images\image-20250109193315059.png)

![image-20250109193342067](C:\Users\86139\AppData\Roaming\Typora\typora-user-images\image-20250109193342067.png)

## Lora(无deepspeed--周锦辉)

![image-20250109193633189](C:\Users\86139\AppData\Roaming\Typora\typora-user-images\image-20250109193633189.png)

![image-20250109193721835](C:\Users\86139\AppData\Roaming\Typora\typora-user-images\image-20250109193721835.png)

![image-20250109193950756](C:\Users\86139\AppData\Roaming\Typora\typora-user-images\image-20250109193950756.png)



![image-20250109201915630](C:\Users\86139\AppData\Roaming\Typora\typora-user-images\image-20250109201915630.png)

### Lora+deepspeed stage 3:

![image-20250109195043112](C:\Users\86139\AppData\Roaming\Typora\typora-user-images\image-20250109195043112.png)

![image-20250109195136991](C:\Users\86139\AppData\Roaming\Typora\typora-user-images\image-20250109195136991.png)

![image-20250109195338978](C:\Users\86139\AppData\Roaming\Typora\typora-user-images\image-20250109195338978.png)

![image-20250109195403484](C:\Users\86139\AppData\Roaming\Typora\typora-user-images\image-20250109195403484.png)

![image-20250109195525153](C:\Users\86139\AppData\Roaming\Typora\typora-user-images\image-20250109195525153.png)

### ![image-20250109200733857](C:\Users\86139\AppData\Roaming\Typora\typora-user-images\image-20250109200733857.png)

![image-20250109201525717](C:\Users\86139\AppData\Roaming\Typora\typora-user-images\image-20250109201525717.png)



### LoRA--Stage 2

###  

![image-20250109201105905](C:\Users\86139\AppData\Roaming\Typora\typora-user-images\image-20250109201105905.png)

![image-20250109201121640](C:\Users\86139\AppData\Roaming\Typora\typora-user-images\image-20250109201121640.png)

![image-20250109201316346](C:\Users\86139\AppData\Roaming\Typora\typora-user-images\image-20250109201316346.png)

![image-20250109201722711](C:\Users\86139\AppData\Roaming\Typora\typora-user-images\image-20250109201722711.png)