# MiniOneRec 复现增量学习笔记

> 基于8卡3090实际复现过程中的学习与发现

---

## 一、硬件适配相关学习点

### 1.1 显存计算与规划

**学习点：** 如何精确计算LLM训练的显存需求

```
【公式】
显存 = 模型参数 + 优化器状态 + 梯度 + 激活值

【详细计算】
- 模型参数：参数量 × 每参数字节数（bf16=2字节）
- 优化器状态（Adam）：参数量 × 8字节（m和v各4字节）
- 梯度：参数量 × 2字节（bf16）
- 激活值：取决于batch_size、序列长度、模型结构

【示例：Qwen2.5-7B】
- 模型参数：7B × 2 = 14GB
- 优化器状态：7B × 8 = 56GB
- 梯度：7B × 2 = 14GB
- 激活值：约10-20GB
- 总计：约100GB
```

**对比之前理解：**
- 之前：只知道"大模型需要大显存"
- 现在：能精确计算每个部分的显存占用，知道优化器状态是大头

**实践验证：**
```bash
# 实际监控显存使用
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

---

### 1.2 DeepSpeed ZeRO各阶段对比

**学习点：** ZeRO-1/2/3的区别和适用场景

| 阶段 | 分摊内容 | 通信开销 | 显存节省 | 适用场景 |
|------|----------|----------|----------|----------|
| ZeRO-1 | 优化器状态 | 低 | ~4x | 显存稍紧 |
| ZeRO-2 | 优化器状态 + 梯度 | 中 | ~8x | 显存较紧 |
| ZeRO-3 | 优化器状态 + 梯度 + 参数 | 高 | ~Nx | 显存非常紧 |

**对比之前理解：**
- 之前：只知道ZeRO能省显存
- 现在：理解了每个阶段省的是什么，通信开销如何权衡

**代码中的体现：**
```yaml
# config/zero2_opt.yaml
deepspeed_config:
  zero_stage: 2  # 分摊优化器状态和梯度
  offload_optimizer_device: none  # 不offload到CPU
  offload_param_device: none
```

---

### 1.3 Gradient Checkpointing原理

**学习点：** 用时间换空间的策略

```
【原理】
正常训练：保存所有层的激活值，用于反向传播
Gradient Checkpointing：只保存部分层的激活值，其他层在反向传播时重新计算

【权衡】
- 显存节省：约50-70%
- 训练时间增加：约20-30%

【适用场景】
- 显存紧张但时间充裕
- 序列较长导致激活值占用大
```

**代码实现：**
```python
model.gradient_checkpointing_enable()
```

---

## 二、训练策略学习点

### 2.1 Batch Size与Gradient Accumulation

**学习点：** 有效batch size的计算和调整策略

```
【公式】
有效batch_size = micro_batch_size × gradient_accumulation_steps × num_gpus

【示例】
原始配置：
- batch_size=1024, micro_batch_size=16, 8卡
- gradient_accumulation_steps = 1024 / (16 × 8) = 8

3090适配：
- batch_size=128, micro_batch_size=2, 8卡
- gradient_accumulation_steps = 128 / (2 × 8) = 8
```

**对比之前理解：**
- 之前：batch_size越大越好
- 现在：需要在显存、训练稳定性、收敛速度之间权衡

**实践经验：**
- 有效batch_size太小：梯度估计不准，训练不稳定
- 有效batch_size太大：显存不够，收敛可能变慢
- 推荐从128-256开始尝试

---

### 2.2 学习率调整策略

**学习点：** 不同阶段的学习率设置

```
【SFT阶段】
- 初始学习率：3e-4（全参数微调）或 1e-4（LoRA）
- Warmup：20-100步
- 调度器：Cosine Annealing
- 早停：patience=3

【RL阶段】
- 初始学习率：1e-5 到 1e-6（比SFT小10-100倍）
- Warmup：3%的总步数
- 调度器：Cosine
- 原因：RL训练更不稳定，需要小学习率
```

**对比之前理解：**
- 之前：学习率就是个超参数，调大调小试试
- 现在：理解了为什么RL学习率要小，warmup的作用是什么

---

### 2.3 梯度裁剪的作用

**学习点：** 防止梯度爆炸的关键技术

```python
# 在GRPOConfig中
max_grad_norm = 0.3  # 梯度裁剪阈值

# 原理
if gradient_norm > max_grad_norm:
    gradient = gradient * (max_grad_norm / gradient_norm)
```

**为什么RL阶段需要更严格的裁剪：**
- RL的奖励信号可能有极端值
- 组内归一化后advantage可能很大
- 不裁剪容易导致训练崩溃

---

## 三、RL相关学习点

### 3.1 GRPO算法细节

**学习点：** GRPO的完整流程

```
【完整流程】
1. 采样：对每个prompt生成G个候选
2. 计算奖励：用reward function评估每个候选
3. 组内归一化：advantage = (reward - mean) / std
4. 计算ratio：π(a|s) / π_old(a|s)
5. 计算loss：clip(ratio) × advantage + β × KL
6. 更新参数

【关键公式】
- ratio = exp(log_π - log_π_old)
- KL = exp(log_π_ref - log_π) - (log_π_ref - log_π) - 1
- loss = -(min(ratio×advantage, clip(ratio)×advantage) - β×KL)
```

**对比之前理解：**
- 之前：知道GRPO是PPO的变体
- 现在：理解了每一步的具体计算，特别是KL惩罚的作用

---

### 3.2 奖励函数设计

**学习点：** 不同奖励函数的适用场景

```
【规则奖励】
- 优点：简单、确定性
- 缺点：稀疏（只有0/1）、无法区分好坏程度
- 适用：冷启动、快速验证

【排序奖励】
- 优点：考虑位置信息
- 缺点：需要预定义位置权重
- 适用：推荐场景

【语义奖励】
- 优点：稠密、考虑相似度
- 缺点：需要额外embedding模型
- 适用：有好的embedding模型时

【协同过滤奖励】
- 优点：利用用户行为模式
- 缺点：需要训练额外模型
- 适用：有足够的交互数据
```

**代码中的实现：**
```python
# rl.py中的奖励选择
if reward_type == "rule":
    reward_fun = rule_reward
elif reward_type == "ranking":
    reward_fun = [rule_reward, ndcg_rule_reward]  # 组合奖励
elif reward_type == "sasrec":
    reward_fun = cf_reward
```

---

### 3.3 约束解码的实现

**学习点：** 如何保证生成的SID有效

```
【实现步骤】
1. 预处理：遍历所有物品SID，构建前缀→合法token的映射
2. 生成时：每一步查找当前前缀的合法token，将其他token设为-inf

【数据结构】
hash_dict = {
    prefix_hash: [valid_token_1, valid_token_2, ...]
}

【时间复杂度】
- 预处理：O(N × L)，N=物品数，L=SID长度
- 生成时：O(1)哈希查找
```

**对比之前理解：**
- 之前：知道要约束解码，不知道怎么实现
- 现在：理解了前缀树+哈希的实现方式

---

## 四、工程实践学习点

### 4.1 多卡训练的通信

**学习点：** NCCL通信问题排查

```
【常见问题】
1. NCCL_IB_DISABLE=1：禁用InfiniBand，解决某些集群的通信问题
2. 主端口冲突：--main_process_port 29503
3. 网络接口：NCCL_SOCKET_IFNAME=eth0

【排查命令】
# 测试NCCL通信
python -c "import torch; torch.distributed.init_process_group('nccl'); print('OK')"

# 查看NCCL日志
export NCCL_DEBUG=INFO
```

---

### 4.2 DeepSpeed配置调优

**学习点：** 关键参数的含义

```yaml
deepspeed_config:
  # 桶大小：控制通信粒度
  reduce_bucket_size: 5e8  # 梯度规约的桶大小
  allgather_bucket_size: 5e8  # 参数收集的桶大小
  
  # 通信优化
  overlap_comm: true  # 通信和计算重叠
  contiguous_gradients: true  # 连续内存，减少碎片
  
  # CPU offload（显存不够时使用）
  offload_optimizer_device: cpu  # 优化器状态放到CPU
  offload_param_device: cpu  # 模型参数放到CPU
```

---

### 4.3 训练监控

**学习点：** 如何有效监控训练过程

```
【关键指标】
1. Loss曲线：是否收敛、是否震荡
2. 梯度norm：是否爆炸、是否消失
3. 学习率：是否按预期变化
4. GPU显存：是否有泄漏
5. 训练速度：是否有瓶颈

【监控工具】
- wandb：可视化训练过程
- nvidia-smi：GPU使用情况
- torch.profiler：性能分析

【告警设置】
- loss突然变大：可能是学习率问题
- 梯度norm突然变大：可能是数据问题
- 显存持续增长：可能是内存泄漏
```

---

## 五、评估相关学习点

### 5.1 推荐指标理解

**学习点：** HR@K和NDCG@K的计算

```
【HR@K (Hit Rate)】
- 含义：在Top-K推荐中，命中目标物品的比例
- 计算：HR@K = (命中次数) / (总样本数)
- 特点：只关心是否命中，不关心位置

【NDCG@K (Normalized Discounted Cumulative Gain)】
- 含义：考虑位置的命中率
- 计算：NDCG@K = (1/log2(rank+1)) / (1/log2(2))
- 特点：位置越靠前，权重越大

【CC (Collision Count)】
- 含义：生成的无效物品数量
- 计算：预测结果不在物品库中的数量
- 用途：验证约束解码是否生效
```

---

### 5.2 评估流程优化

**学习点：** 多卡并行评估

```bash
# 1. 拆分测试数据
python split.py --input_path test.csv --output_path ./temp --cuda_list "0,1,2,3,4,5,6,7"

# 2. 并行评估
for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i python evaluate.py ... &
done
wait

# 3. 合并结果
python merge.py --input_path ./temp --output_path ./results
```

**对比之前理解：**
- 之前：评估就是跑一遍
- 现在：理解了如何并行化评估，如何拆分和合并

---

## 六、代码结构理解

### 6.1 模块职责划分

```
【数据模块】
- data.py: 定义各种Dataset类
- convert_dataset.py: 数据格式转换
- data/amazon18_data_process.py: 原始数据预处理

【训练模块】
- sft.py: SFT训练主循环
- rl.py: RL训练主循环
- minionerec_trainer.py: 自定义GRPO Trainer

【模型模块】
- rq/rqvae.py: RQ-VAE训练
- sasrec.py: SASRec模型（用于CF奖励）

【工具模块】
- LogitProcessor.py: 约束解码器
- evaluate.py: 评估脚本
- calc.py: 指标计算
```

---

### 6.2 数据流理解

```
【SFT数据流】
原始CSV → data.py中的Dataset → DataLoader → Trainer

【RL数据流】
原始CSV → SidDataset → Dataset.from_dict → ReReTrainer
                                              ↓
                                         生成候选 → 计算奖励 → 更新策略

【评估数据流】
测试CSV → EvalSidDataset → 模型生成 → decode → 计算指标
```

---

## 七、问题排查经验

### 7.1 OOM问题排查

```
【排查步骤】
1. 确认显存占用：nvidia-smi
2. 减小batch_size：micro_batch_size 2 → 1
3. 减小序列长度：cutoff_len 256 → 128
4. 启用gradient checkpointing
5. 使用ZeRO-3 + CPU offload

【经验】
- 模型参数占显存大头
- 优化器状态是第二大户
- 激活值随batch_size和序列长度线性增长
```

---

### 7.2 训练不收敛排查

```
【排查步骤】
1. 检查loss曲线：是否在下降
2. 检查学习率：是否在合理范围
3. 检查梯度norm：是否爆炸/消失
4. 检查数据：是否有异常值
5. 检查模型：是否正确加载

【常见原因】
- 学习率太大：loss震荡或发散
- 学习率太小：loss下降很慢
- 数据问题：loss突然变大
- 模型问题：loss不下降
```

---

### 7.3 约束解码失败排查

```
【症状】
CC > 0，说明生成了无效物品

【排查步骤】
1. 检查.index.json格式：是否正确
2. 检查token映射：是否和模型词表一致
3. 检查transformers版本：不同版本可能有差异
4. 检查模型类型：Instruct模型可能有问题

【解决方案】
- 使用base模型（非Instruct）
- 检查并修复token映射
- 更新transformers版本
```

---

## 八、优化技巧总结

### 8.1 显存优化

```
1. 使用小模型：7B → 3B → 1.5B
2. 减小batch_size：micro_batch_size 16 → 2
3. 减小序列长度：512 → 256 → 128
4. 使用gradient checkpointing
5. 使用ZeRO-3 + CPU offload
6. 使用LoRA/QLoRA
```

---

### 8.2 速度优化

```
1. 增大micro_batch_size（显存允许的情况下）
2. 使用torch.compile
3. 使用Flash Attention
4. 减少数据加载的num_workers
5. 使用NVMe SSD
6. 使用混合精度训练（bf16）
```

---

### 8.3 效果优化

```
1. 调整学习率和warmup
2. 调整batch_size
3. 调整奖励函数设计
4. 调整num_generations
5. 尝试不同模型大小
6. 使用多任务训练
```

---

## 九、与之前知识的对比总结

| 主题 | 之前理解 | 现在理解 |
|------|----------|----------|
| 显存计算 | 大模型需要大显存 | 能精确计算各部分占用 |
| DeepSpeed | ZeRO能省显存 | 理解1/2/3的区别和适用场景 |
| Batch Size | 越大越好 | 需要权衡显存、稳定性、速度 |
| 学习率 | 调参试试 | 理解不同阶段的设置策略 |
| GRPO | PPO变体 | 理解完整流程和每步计算 |
| 约束解码 | 保证有效生成 | 理解前缀树+哈希实现 |
| 推荐指标 | HR、NDCG | 理解计算方法和物理含义 |
| 多卡训练 | torchrun启动 | 理解NCCL通信和配置调优 |

---

## 十、复现过程中的关键发现

### 10.1 小模型也能工作

**发现：** 1.5B模型在Industrial数据集上也能达到不错的效果

**原因分析：**
- 数据集较小（36K样本），大模型可能过拟合
- 推荐任务不需要太强的推理能力
- SID已经压缩了物品信息，降低了任务难度

---

### 10.2 RL阶段的显存瓶颈

**发现：** RL阶段比SFT更耗显存

**原因分析：**
- 需要同时加载策略模型和参考模型
- 需要生成多个候选（num_generations）
- 候选的logits和梯度都需要保存

**解决方案：**
- 减小num_generations：16 → 8
- 使用sync_ref_model共享参数
- 使用beam_search减少采样开销

---

### 10.3 约束解码的重要性

**发现：** 不使用约束解码时，CC很高，效果很差

**原因分析：**
- LLM词表很大（32K），但有效SID只有几千个
- 不约束时，大部分采样都是无效的
- 无效采样浪费计算资源，降低训练效率

**验证方法：**
```bash
# 对比有无约束解码的效果
# 有约束：CC=0，效果好
# 无约束：CC>0，效果差
```

---

## 十一、学习资源推荐

### 11.1 DeepSpeed官方文档
- https://www.deepspeed.ai/docs/config-json/
- 理解ZeRO各阶段的原理和配置

### 11.2 TRL库文档
- https://huggingface.co/docs/trl/
- 理解GRPO的实现细节

### 11.3 推荐系统论文
- MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation
- GRPO: DeepSeekMath: Pushing the Limits of Mathematical Reasoning

### 11.4 代码阅读建议
1. 先看data.py理解数据格式
2. 再看sft.py理解SFT流程
3. 然后看minionerec_trainer.py理解GRPO实现
4. 最后看rl.py理解RL流程

---

**持续更新中...**
