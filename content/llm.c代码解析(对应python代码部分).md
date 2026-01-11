
# Python 实现 GPT-2 训练（对照 train_gpt2.c）


本文档详细对比 **train_gpt2.c** 和 **Python 实现**之间的对应关系。

---


## 📦 文件说明

我为你创建了两个 Python 实现版本：
  

1. **train_gpt2_simple.py** - 使用 PyTorch（推荐，自动微分）

2. **train_gpt2_numpy.py** - 使用纯 NumPy（教育性强，手动实现所有循环）

---
## 🔧 环境准备

### 安装依赖

```bash

# 安装 PyTorch 版本（推荐）

pip install torch numpy

  

# 或者只安装 NumPy（纯 NumPy 版本）

pip install numpy

```

  

### 运行代码

  

```bash

# PyTorch 版本

python3 train_gpt2_simple.py

  

# NumPy 版本（较慢但教育性强）

python3 train_gpt2_numpy.py

```

  

---

  

## 📊 代码对比：C vs Python

  

### 1️⃣ **配置结构**

  

#### C 代码

```c

typedef struct {

int max_seq_len;

int vocab_size;

int padded_vocab_size;

int num_layers;

int num_heads;

int channels;

} GPT2Config;

```

  

#### Python (PyTorch)

```python

@dataclass

class GPT2Config:

max_seq_len: int = 1024

vocab_size: int = 50257

padded_vocab_size: int = 50304

num_layers: int = 12

num_heads: int = 12

channels: int = 768

```

  

---

  

### 2️⃣ **Encoder（Token + 位置嵌入）**

  

#### C 代码

```c

void encoder_forward(float* out, int* inp, float* wte, float* wpe,

int B, int T, int C) {

for (int b = 0; b < B; b++) {

for (int t = 0; t < T; t++) {

float* out_bt = out + b * T * C + t * C;

int ix = inp[b * T + t];

float* wte_ix = wte + ix * C;

float* wpe_t = wpe + t * C;

for (int i = 0; i < C; i++) {

out_bt[i] = wte_ix[i] + wpe_t[i];

}

}

}

}

```

  

#### Python (PyTorch) - 向量化

```python

def encoder_forward(inp: torch.Tensor, wte: torch.Tensor, wpe: torch.Tensor):

"""

inp: (B, T) token IDs

wte: (V, C) token embeddings

wpe: (maxT, C) position embeddings

返回: (B, T, C)

"""

B, T = inp.shape

token_embeddings = wte[inp] # (B, T, C)

positions = torch.arange(T, device=inp.device)

position_embeddings = wpe[positions] # (T, C)

return token_embeddings + position_embeddings # 广播

```

  

#### Python (NumPy) - 显式循环

```python

def encoder_forward(inp: np.ndarray, wte: np.ndarray, wpe: np.ndarray):

B, T = inp.shape

C = wte.shape[1]

out = np.zeros((B, T, C), dtype=np.float32)

# 与 C 代码完全对应的循环

for b in range(B):

for t in range(T):

ix = inp[b, t]

out[b, t, :] = wte[ix, :] + wpe[t, :]

return out

```

  

**对应关系：**

- C 的指针偏移 → Python 的数组索引

- C 的三层循环 → PyTorch 的向量化 / NumPy 的显式循环

  

---

  

### 3️⃣ **LayerNorm（层归一化）**

  

#### C 代码

```c

void layernorm_forward(float* out, float* mean, float* rstd,

float* inp, float* weight, float* bias,

int B, int T, int C) {

float eps = 1e-5f;

for (int b = 0; b < B; b++) {

for (int t = 0; t < T; t++) {

float* x = inp + b * T * C + t * C;

// 计算均值

float m = 0.0f;

for (int i = 0; i < C; i++) {

m += x[i];

}

m = m/C;

// 计算方差

float v = 0.0f;

for (int i = 0; i < C; i++) {

float xshift = x[i] - m;

v += xshift * xshift;

}

v = v/C;

// 归一化

float s = 1.0f / sqrtf(v + eps);

float* out_bt = out + b * T * C + t * C;

for (int i = 0; i < C; i++) {

float n = (s * (x[i] - m));

float o = n * weight[i] + bias[i];

out_bt[i] = o;

}

mean[b * T + t] = m;

rstd[b * T + t] = s;

}

}

}

```

  

#### Python (PyTorch)

```python

def layernorm_forward(inp, weight, bias, eps=1e-5):

"""inp: (B, T, C)"""

mean = inp.mean(dim=-1, keepdim=True) # (B, T, 1)

var = inp.var(dim=-1, keepdim=True, unbiased=False) # (B, T, 1)

rstd = 1.0 / torch.sqrt(var + eps) # (B, T, 1)

norm = (inp - mean) * rstd # (B, T, C)

out = norm * weight + bias # (B, T, C)

return out, mean.squeeze(-1), rstd.squeeze(-1)

```

  

#### Python (NumPy)

```python

def layernorm_forward(inp, weight, bias, eps=1e-5):

B, T, C = inp.shape

mean = np.mean(inp, axis=-1) # (B, T)

var = np.var(inp, axis=-1) # (B, T)

rstd = 1.0 / np.sqrt(var + eps) # (B, T)

out = np.zeros_like(inp)

# 与 C 代码完全对应的循环

for b in range(B):

for t in range(T):

norm = (inp[b, t, :] - mean[b, t]) * rstd[b, t]

out[b, t, :] = norm * weight + bias

return out, mean, rstd

```

  

---

  

### 4️⃣ **Attention（自注意力机制）**

#### C 代码（简化）

```c

void attention_forward(float* out, float* preatt, float* att,

float* inp, int B, int T, int C, int NH) {

int hs = C / NH;

float scale = 1.0 / sqrtf(hs);

for (int b = 0; b < B; b++) {

for (int t = 0; t < T; t++) {

for (int h = 0; h < NH; h++) {

// Pass 1: Q @ K

for (int t2 = 0; t2 <= t; t2++) {

float val = 0.0f;

for (int i = 0; i < hs; i++) {

val += query_t[i] * key_t2[i];

}

preatt[t2] = val * scale;

}

// Pass 2 & 3: Softmax

float maxval = max(preatt);

float expsum = sum(exp(preatt - maxval));

for (int t2 = 0; t2 <= t; t2++) {

att[t2] = exp(preatt[t2] - maxval) / expsum;

}

// Pass 4: att @ V

for (int t2 = 0; t2 <= t; t2++) {

for (int i = 0; i < hs; i++) {

out[i] += att[t2] * value_t2[i];

}

}

}

}

}

}

```

  

#### Python (PyTorch) - 使用 scaled_dot_product_attention

```python

def attention_forward(inp, B, T, C, NH):

"""inp: (B, T, 3*C) 包含 Q, K, V"""

qkv = inp.view(B, T, 3, C)

q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

hs = C // NH

q = q.view(B, T, NH, hs).transpose(1, 2) # (B, NH, T, hs)

k = k.view(B, T, NH, hs).transpose(1, 2)

v = v.view(B, T, NH, hs).transpose(1, 2)

# 计算注意力

scale = 1.0 / np.sqrt(hs)

att = (q @ k.transpose(-2, -1)) * scale # (B, NH, T, T)

# 因果掩码

mask = torch.tril(torch.ones(T, T))

att = att.masked_fill(mask == 0, float('-inf'))

# Softmax + 加权求和

att = F.softmax(att, dim=-1)

out = att @ v # (B, NH, T, hs)

return out.transpose(1, 2).contiguous().view(B, T, C)

```

  

---

  

### 5️⃣ **完整的前向传播**

  

#### C 代码

```c

void gpt2_forward(GPT2 *model, int* inputs, int* targets, size_t B, size_t T) {

// 1. Embedding

encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C);

// 2. Transformer 层

for (int l = 0; l < L; l++) {

layernorm_forward(l_ln1, ...);

matmul_forward(l_qkv, ...);

attention_forward(l_atty, ...);

matmul_forward(l_attproj, ...);

residual_forward(l_residual2, ...);

layernorm_forward(l_ln2, ...);

matmul_forward(l_fch, ...);

gelu_forward(l_fch_gelu, ...);

matmul_forward(l_fcproj, ...);

residual_forward(l_residual3, ...);

}

// 3. 输出层

layernorm_forward(acts.lnf, ...);

matmul_forward(acts.logits, ...);

softmax_forward(acts.probs, ...);

// 4. 损失

if (targets != NULL) {

crossentropy_forward(model->acts.losses, ...);

}

}

```

  

#### Python (PyTorch)

```python

class GPT2(nn.Module):

def forward(self, inputs, targets=None):

B, T = inputs.shape

# 1. Embedding

x = encoder_forward(inputs, self.wte, self.wpe)

# 2. Transformer 层

residual = x

for l in range(self.config.num_layers):

# Pre-LN + Attention

ln1_out, _, _ = layernorm_forward(residual, self.ln1w[l], self.ln1b[l])

qkv = matmul_forward(ln1_out, self.qkvw[l], self.qkvb[l])

atty = attention_forward(qkv, B, T, C, NH)

attproj = matmul_forward(atty, self.attprojw[l], self.attprojb[l])

residual2 = residual + attproj

# Pre-LN + MLP

ln2_out, _, _ = layernorm_forward(residual2, self.ln2w[l], self.ln2b[l])

fch = matmul_forward(ln2_out, self.fcw[l], self.fcb[l])

fch_gelu = gelu_forward(fch)

fcproj = matmul_forward(fch_gelu, self.fcprojw[l], self.fcprojb[l])

residual = residual2 + fcproj

# 3. 输出层

lnf_out, _, _ = layernorm_forward(residual, self.lnfw, self.lnfb)

logits = matmul_forward(lnf_out, self.wte, None) # 权重共享

# 4. 损失

loss = None

if targets is not None:

probs = softmax_forward(logits)

losses = crossentropy_forward(probs, targets)

loss = losses.mean()

return logits, loss

```

  

---

  

### 6️⃣ **训练循环**

  

#### C 代码

```c

int main() {

GPT2 model;

gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

for (int step = 0; step <= 40; step++) {

// 训练一步

dataloader_next_batch(&train_loader);

gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);

gpt2_zero_grad(&model);

gpt2_backward(&model);

gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);

printf("step %d: train loss %f\n", step, model.mean_loss);

}

}

```

  

#### Python (PyTorch)

```python

def main():

config = GPT2Config(...)

model = GPT2(config)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for step in range(40):

# 生成/加载数据

inputs = ...

targets = ...

# 前向传播

logits, loss = model(inputs, targets)

# 反向传播

optimizer.zero_grad()

loss.backward() # PyTorch 自动求导！

optimizer.step()

print(f"step {step}: train loss {loss.item():.6f}")

```

  

---

  

## 🎯 关键对应关系总结

  

| C 代码 | Python (PyTorch) | Python (NumPy) |

|--------|-----------------|----------------|

| `float*` 指针 | `torch.Tensor` | `np.ndarray` |

| 手动内存管理 | 自动内存管理 | 自动内存管理 |

| 多层 for 循环 | 向量化操作 | 显式 for 循环 |

| 手写梯度计算 | `loss.backward()` | 需手写（未实现） |

| `malloc/free` | 自动垃圾回收 | 自动垃圾回收 |

| OpenMP 并行 | CUDA 并行 | 单线程 |

  

---

  

## 🚀 优缺点对比

  

### C 版本（train_gpt2.c）

✅ **优点：**

- 完全控制内存和性能

- 教育性强，每步都清晰可见

- 无依赖，可移植性强

  

❌ **缺点：**

- 代码量大，需手写所有操作

- 手动管理内存容易出错

- 调试困难

  

### Python + PyTorch

✅ **优点：**

- 代码简洁，易于实验

- 自动求导，无需手写梯度

- GPU 加速开箱即用

- 丰富的生态系统

  

❌ **缺点：**

- 黑盒操作，不利于理解底层

- 依赖庞大（PyTorch ~1GB）

- 性能不如手工优化的 C/CUDA

  

### Python + NumPy

✅ **优点：**

- 与 C 代码结构一一对应

- 教育性强，易于理解

- 依赖小，只需 NumPy

  

❌ **缺点：**

- 速度慢（纯 CPU，无并行）

- 不适合实际训练

  

---

  

## 📝 学习建议

  

1. **先读 C 代码** - 理解每个操作的底层实现

2. **对照 NumPy 版本** - 看 Python 如何映射 C 的循环结构

3. **学习 PyTorch 版本** - 理解如何用高级 API 实现同样的功能

4. **实验修改** - 尝试改变模型结构，观察变化

  

---

  

## 🔗 运行示例

  

```bash

# 如果你的环境没有安装依赖，先安装：

pip install torch numpy

  

# 运行 PyTorch 版本（推荐）

python3 train_gpt2_simple.py

  

# 运行 NumPy 版本（教育性强但较慢）

python3 train_gpt2_numpy.py

  

# 对比 C 版本

make train_gpt2

OMP_NUM_THREADS=8 ./train_gpt2

```

  

现在你有了三个版本的代码可以对比学习！🎉