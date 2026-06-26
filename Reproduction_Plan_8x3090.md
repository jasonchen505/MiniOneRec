# MiniOneRec 8卡3090复现计划

> 硬件资源：8 × NVIDIA RTX 3090 (24GB)
> 原始需求：4-8 × A100/H100 (80GB)

---

## 一、资源评估与可行性分析

### 1.1 硬件对比

| 项目 | 原始需求 | 实际资源 | 差距 |
|------|----------|----------|------|
| GPU数量 | 4-8张 | 8张 | 满足 |
| 单卡显存 | 80GB | 24GB | 3.3倍 |
| 总显存 | 320-640GB | 192GB | 1.7-3.3倍 |
| GPU架构 | A100/H100 | 3090 | 较老 |

### 1.2 数据规模

| 数据集 | 训练样本 | 物品数量 | SID格式 |
|--------|----------|----------|---------|
| Industrial_and_Scientific | 36,260 | 3,686 | 3层 `<a_X><b_Y><c_Z>` |
| Office_Products | 38,925 | ~4,000 | 3层 |

### 1.3 显存需求估算

**SFT阶段（以Qwen2.5-7B为例）：**
- 模型参数：7B × 2 bytes (bf16) = 14GB
- 优化器状态（Adam）：7B × 8 bytes = 56GB
- 梯度：7B × 2 bytes = 14GB
- 激活值：约10-20GB（取决于序列长度）
- **总计：约100GB**

**使用DeepSpeed ZeRO-2（8卡分摊）：**
- 每卡：14GB (模型) + 7GB (优化器/8) + 7GB (梯度/8) + 10GB (激活) ≈ 38GB
- **超出3090的24GB，需要进一步优化**

**RL阶段额外开销：**
- 参考模型：额外14GB
- 生成16个候选：额外显存
- **更加紧张**

### 1.4 可行性结论

**完全复现（7B模型）**：❌ 不可行，显存不足

**可行方案：**
1. ✅ 使用小模型（1.5B-3B）
2. ✅ 使用LoRA/QLoRA减少训练参数
3. ✅ 使用ZeRO-3 + CPU offload
4. ✅ 减小batch_size和num_generations

---

## 二、复现策略选择

### 方案A：使用小模型（推荐）

| 项目 | 配置 |
|------|------|
| 基础模型 | Qwen2.5-1.5B 或 Qwen2.5-3B |
| 训练方式 | 全参数微调 |
| DeepSpeed | ZeRO-2 |
| 预计显存 | ~18-22GB/卡 |

**优点：** 实现简单，训练速度快
**缺点：** 效果可能不如7B模型

### 方案B：使用LoRA微调大模型

| 项目 | 配置 |
|------|------|
| 基础模型 | Qwen2.5-7B |
| 训练方式 | LoRA (rank=8-16) |
| DeepSpeed | ZeRO-2 + gradient checkpointing |
| 预计显存 | ~20-24GB/卡 |

**优点：** 可以用大模型
**缺点：** 实现复杂，需要修改代码

### 方案C：使用ZeRO-3 + Offload

| 项目 | 配置 |
|------|------|
| 基础模型 | Qwen2.5-7B |
| 训练方式 | 全参数微调 |
| DeepSpeed | ZeRO-3 + CPU offload |
| 预计显存 | ~16-20GB/卡 |

**优点：** 可以用大模型和全参数
**缺点：** 训练速度慢（CPU offload开销大）

### 推荐方案：方案A

**理由：**
1. 实现最简单，代码改动最小
2. Industrial数据集较小（36K样本），小模型可能足够
3. 训练速度快，便于快速迭代验证
4. 如果效果不够，再尝试方案B

---

## 三、详细复现计划

### Phase 0: 环境准备（1天）

#### 0.1 创建环境
```bash
conda create -n MiniOneRec python=3.11 -y
conda activate MiniOneRec
pip install -r requirements.txt
```

#### 0.2 验证GPU
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
```

#### 0.3 下载模型
```bash
# 方案A: 使用小模型
# 下载 Qwen2.5-1.5B 或 Qwen2.5-3B
huggingface-cli download Qwen/Qwen2.5-1.5B --local-dir ./models/qwen2.5-1.5b

# 或者使用镜像
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen2.5-1.5B --local-dir ./models/qwen2.5-1.5b
```

#### 0.4 验证数据
```bash
ls -la data/Amazon/train/
ls -la data/Amazon/index/
ls -la data/Amazon/info/
```

---

### Phase 1: SID构建（跳过，使用预构建）

**说明：** 项目已提供预构建的SID，直接使用即可

```bash
# 已有文件：
# data/Amazon/index/Industrial_and_Scientific.index.json
# data/Amazon/index/Industrial_and_Scientific.item.json
# data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy
```

**如果需要重新构建SID（可选）：**
```bash
# 1. 训练RQ-VAE
bash rq/rqvae.sh \
    --data_path ./data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy \
    --ckpt_dir ./output/rqvae \
    --lr 1e-3 \
    --epochs 5000 \
    --batch_size 2048 \
    --num_emb_list 256 256 256

# 2. 生成索引
python rq/generate_indices.py
```

---

### Phase 2: SFT训练（2-3天）

#### 2.1 修改配置

创建新的sft脚本 `sft_3090.sh`：

```bash
#!/bin/bash
export NCCL_IB_DISABLE=1

category="Industrial_and_Scientific"
train_file=$(ls -f ./data/Amazon/train/${category}*11.csv)
eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
info_file=$(ls -f ./data/Amazon/info/${category}*.txt)

# 关键修改：调整参数适配3090
torchrun --nproc_per_node 8 \
    sft.py \
    --base_model ./models/qwen2.5-1.5b \
    --batch_size 128 \
    --micro_batch_size 2 \
    --train_file ${train_file} \
    --eval_file ${eval_file} \
    --output_dir ./output/sft_1.5b \
    --category ${category} \
    --train_from_scratch False \
    --seed 42 \
    --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
    --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json \
    --freeze_LLM False \
    --num_epochs 10 \
    --learning_rate 3e-4 \
    --cutoff_len 256
```

#### 2.2 关键参数调整说明

| 参数 | 原始值 | 调整值 | 原因 |
|------|--------|--------|------|
| base_model | Qwen2.5-7B | Qwen2.5-1.5B | 减少显存占用 |
| batch_size | 1024 | 128 | 减少显存占用 |
| micro_batch_size | 16 | 2 | 适配24GB显存 |
| cutoff_len | 512 | 256 | 减少序列长度 |
| gradient_accumulation_steps | 自动计算 | 8 | 保持有效batch_size |

#### 2.3 显存优化技巧

```python
# 在sft.py中添加gradient checkpointing
model.gradient_checkpointing_enable()

# 或者使用torch.compile优化（PyTorch 2.0+）
# model = torch.compile(model)
```

#### 2.4 启动训练
```bash
bash sft_3090.sh
```

#### 2.5 监控训练
```bash
# 查看GPU使用情况
watch -n 1 nvidia-smi

# 查看训练日志
tail -f output/sft_1.5b/training.log

# 使用wandb监控（可选）
wandb login
```

---

### Phase 3: RL训练（2-3天）

#### 3.1 修改配置

创建新的rl脚本 `rl_3090.sh`：

```bash
#!/bin/bash
export NCCL_IB_DISABLE=1

category="Industrial_and_Scientific"
train_file=$(ls -f ./data/Amazon/train/${category}*.csv)
eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
info_file=$(ls -f ./data/Amazon/info/${category}*.txt)

HF_ENDPOINT=https://hf-mirror.com accelerate launch \
    --config_file ./config/zero2_opt.yaml \
    --num_processes 8 \
    --main_process_port 29503 \
    rl.py \
    --model_path ./output/sft_1.5b/final_checkpoint \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --num_train_epochs 2 \
    --gradient_accumulation_steps 4 \
    --train_file ${train_file} \
    --eval_file ${eval_file} \
    --info_file ${info_file} \
    --category ${category} \
    --sample_train False \
    --eval_step 0.0999 \
    --reward_type rule \
    --num_generations 8 \
    --mask_all_zero False \
    --dynamic_sampling False \
    --sync_ref_model True \
    --beam_search True \
    --test_during_training False \
    --temperature 1.0 \
    --learning_rate 1e-5 \
    --add_gt False \
    --beta 1e-3 \
    --dapo False \
    --output_dir ./output/rl_1.5b \
    --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
    --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json
```

#### 3.2 关键参数调整

| 参数 | 原始值 | 调整值 | 原因 |
|------|--------|--------|------|
| train_batch_size | 64 | 8 | 减少显存占用 |
| eval_batch_size | 128 | 16 | 减少显存占用 |
| gradient_accumulation_steps | 2 | 4 | 保持有效batch_size |
| num_generations | 16 | 8 | 减少生成候选数 |
| reward_type | ranking | rule | 简化奖励计算 |

#### 3.3 显存优化

如果仍然OOM，可以尝试：

```bash
# 1. 使用ZeRO-3
# 修改 config/zero2_opt.yaml 为 zero3_opt.yaml
# zero_stage: 3

# 2. 添加CPU offload
# offload_param_device: cpu
# offload_optimizer_device: cpu

# 3. 进一步减小num_generations
# num_generations: 4
```

#### 3.4 启动训练
```bash
bash rl_3090.sh
```

---

### Phase 4: 评估（半天）

#### 4.1 修改评估脚本

创建 `evaluate_3090.sh`：

```bash
#!/bin/bash
category="Industrial_and_Scientific"
exp_name="./output/rl_1.5b/final_checkpoint"

exp_name_clean=$(basename "$exp_name")
test_file=$(ls ./data/Amazon/test/${category}*11.csv 2>/dev/null | head -1)
info_file=$(ls ./data/Amazon/info/${category}*.txt 2>/dev/null | head -1)

temp_dir="./temp/${category}-${exp_name_clean}"
mkdir -p "$temp_dir"

# 拆分测试数据到8张GPU
python ./split.py --input_path "$test_file" --output_path "$temp_dir" --cuda_list "0,1,2,3,4,5,6,7"

# 并行评估
for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i python -u ./evaluate.py \
        --base_model "$exp_name" \
        --info_file "$info_file" \
        --category ${category} \
        --test_data_path "$temp_dir/${i}.csv" \
        --result_json_data "$temp_dir/${i}.json" \
        --batch_size 4 \
        --num_beams 20 \
        --max_new_tokens 128 &
done
wait

# 合并结果
python ./merge.py \
    --input_path "$temp_dir" \
    --output_path "./results/${exp_name_clean}/final_result_${category}.json" \
    --cuda_list "0,1,2,3,4,5,6,7"

# 计算指标
python ./calc.py \
    --path "./results/${exp_name_clean}/final_result_${category}.json" \
    --item_path "$info_file"
```

#### 4.2 启动评估
```bash
bash evaluate_3090.sh
```

---

## 四、预期结果与对比

### 4.1 预期指标范围

| 指标 | 7B模型（原始） | 1.5B模型（预期） | 说明 |
|------|----------------|------------------|------|
| HR@10 | ~0.xxxx | ~0.xxxx | 小模型可能略低 |
| NDCG@10 | ~0.xxxx | ~0.xxxx | 小模型可能略低 |
| CC | 0 | 0 | 约束解码应该生效 |

### 4.2 训练时间预估

| 阶段 | 7B模型（8×A100） | 1.5B模型（8×3090） |
|------|------------------|---------------------|
| SFT | ~6小时 | ~4-6小时 |
| RL | ~12小时 | ~8-12小时 |
| 评估 | ~1小时 | ~1小时 |
| **总计** | ~19小时 | ~13-19小时 |

---

## 五、故障排查指南

### 5.1 常见问题

#### 问题1: CUDA OOM
```
RuntimeError: CUDA out of memory
```

**解决方案：**
1. 减小micro_batch_size（2 → 1）
2. 减小cutoff_len（256 → 128）
3. 启用gradient checkpointing
4. 使用ZeRO-3 + CPU offload

#### 问题2: 训练loss为NaN
```
loss: nan
```

**解决方案：**
1. 降低学习率（3e-4 → 1e-4）
2. 增加warmup_steps
3. 检查数据是否有异常值
4. 使用gradient clipping

#### 问题3: 约束解码失败（CC > 0）
```
CC: 1234
```

**解决方案：**
1. 检查transformers版本
2. 切换到base模型（非Instruct）
3. 检查.index.json格式

#### 问题4: 训练速度慢
**解决方案：**
1. 增大micro_batch_size（如果显存允许）
2. 使用torch.compile优化
3. 减少数据加载的num_workers
4. 使用NVMe SSD存储数据

### 5.2 调试技巧

```bash
# 1. 单卡测试
CUDA_VISIBLE_DEVICES=0 python sft.py --micro_batch_size 1 ...

# 2. 查看详细错误
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# 3. 使用torch profiler分析瓶颈
python -m torch.profiler ...

# 4. 监控GPU显存
watch -n 1 nvidia-smi
```

---

## 六、优化进阶（可选）

### 6.1 如果1.5B效果不够好

**尝试1: 使用3B模型**
```bash
--base_model ./models/qwen2.5-3b
--micro_batch_size 1
--cutoff_len 128
```

**尝试2: 使用LoRA微调7B模型**
```python
# 在sft.py中添加LoRA配置
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
```

**尝试3: 使用QLoRA（4bit量化）**
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
)
```

### 6.2 加速训练

**使用torch.compile（PyTorch 2.0+）**
```python
model = torch.compile(model, mode="reduce-overhead")
```

**使用Flash Attention**
```python
# 确保安装了flash-attn
pip install flash-attn --no-build-isolation

# 在模型加载时启用
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)
```

---

## 七、复现检查清单

### Phase 0: 环境准备
- [ ] 创建conda环境
- [ ] 安装依赖
- [ ] 验证GPU可用
- [ ] 下载模型
- [ ] 验证数据文件

### Phase 1: SID构建
- [x] 使用预构建SID（跳过）

### Phase 2: SFT训练
- [ ] 创建sft_3090.sh
- [ ] 单卡测试通过
- [ ] 8卡训练完成
- [ ] 验证模型输出

### Phase 3: RL训练
- [ ] 创建rl_3090.sh
- [ ] 单卡测试通过
- [ ] 8卡训练完成
- [ ] 验证奖励曲线

### Phase 4: 评估
- [ ] 创建evaluate_3090.sh
- [ ] 评估完成
- [ ] 计算指标
- [ ] 记录结果

### 优化（可选）
- [ ] 尝试3B模型
- [ ] 尝试LoRA
- [ ] 尝试QLoRA
- [ ] 调参优化

---

## 八、时间规划

| 天数 | 任务 | 产出 |
|------|------|------|
| Day 1 | 环境准备 + 数据验证 | 可运行的环境 |
| Day 2-3 | SFT训练 | SFT模型 |
| Day 4-5 | RL训练 | RL模型 |
| Day 6 | 评估 + 结果分析 | 指标报告 |
| Day 7 | 优化尝试（可选） | 优化后的模型 |

**总计：6-7天完成完整复现**

---

## 九、资源监控脚本

### GPU监控
```bash
#!/bin/bash
# monitor_gpu.sh
while true; do
    echo "=== $(date) ==="
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv
    sleep 5
done
```

### 训练日志分析
```bash
#!/bin/bash
# analyze_log.sh
LOG_FILE=$1

echo "=== Training Loss ==="
grep "loss" $LOG_FILE | tail -20

echo "=== GPU Memory Usage ==="
grep "memory" $LOG_FILE | tail -10

echo "=== Training Speed ==="
grep "samples_per_second" $LOG_FILE | tail -10
```

---

## 附录：关键配置文件

### config/zero2_opt.yaml（修改版）
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
num_machines: 1
mixed_precision: bf16

deepspeed_config:
  zero_stage: 2
  contiguous_gradients: true
  overlap_comm: true
  reduce_scatter: true
  reduce_bucket_size: 5e8
  allgather_bucket_size: 5e8
  offload_optimizer_device: none
  offload_param_device: none

# 添加gradient checkpointing
gradient_checkpointing: true
gradient_accumulation_steps: 4

debug: false
```

### config/zero3_opt.yaml（备选）
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
num_machines: 1
mixed_precision: bf16

deepspeed_config:
  zero_stage: 3
  contiguous_gradients: true
  overlap_comm: true
  reduce_scatter: true
  reduce_bucket_size: 5e8
  allgather_bucket_size: 5e8
  offload_optimizer_device: cpu
  offload_param_device: cpu

gradient_checkpointing: true
gradient_accumulation_steps: 4

debug: false
```

---

**祝复现顺利！**
