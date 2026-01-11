```YAML
---

mindmap-plugin: basic
display-mode: outline
 
 分布式架构

• 基于Ray的多角色分布式架构

• Actor/Critic/Reference/Reward/Environment Workers

• 灵活的资源分配和异构任务调度

---
```

ROLL代码库功能总结

核心功能
ROLL (Reinforcement Learning Optimization for Large-Scale Learning)
是阿里巴巴开发的大语言模型强化学习训练框架，支持大规模GPU资源调度和训练。
主要功能模块：

1. RLVR Pipeline (多任务RL训练)

• 支持数学、代码、通用推理、开放式问答、指令遵循等多领域训练

• 样本级异步并行Rollout

• 灵活的domain_batch_size分配控制

  

2. Agentic Pipeline (智能体RL)

• 多轮交互能力：游戏、多轮对话、工具使用

• 环境级异步并行rollout和异步训练

• 支持TrajectoryWise (StartPO) 和 StepWise (GiGPO) 训练范式

  

3. 其他Pipeline

• Distill Pipeline（蒸馏）：支持LLM和VLM

• DPO Pipeline（直接偏好优化）

• SFT Pipeline（监督微调）

• Reward FL Pipeline（奖励联邦学习）

  

4. 算法支持

• PPO, GRPO, Reinforce++, TOPR, RAFT++, GSPO, GiGPO, StarPO, Lite

PPO等20+强化学习算法

  

5. 多后端引擎支持

• 推理引擎: vLLM, SGLang（支持FP8动态推理）

• 训练引擎: DeepSpeed (ZeRO), Megatron-Core (5D并行: DP/TP/PP/CP/EP), FSDP

• 特殊支持: LoRA训练, Ascend NPU, AMD GPU

  ```
  
  ```

6. 分布式架构

• 基于Ray的多角色分布式架构

• Actor/Critic/Reference/Reward/Environment Workers

• 灵活的资源分配和异构任务调度

  

7. 可观测性

• 集成SwanLab / WandB / TensorBoard

• 跟踪各领域和奖励类型的性能指标

  

──────────────────────────────────────────

  

目录结构

  
ROLL/

├── roll/ # 核心代码库

│ ├── pipeline/ # 训练管道

│ │ ├── rlvr/ # RLVR管道（多任务RL）

│ │ ├── agentic/ # Agentic管道（智能体RL）

│ │ ├── distill/ # 蒸馏管道

│ │ ├── dpo/ # DPO管道

│ │ ├── sft/ # SFT管道

│ │ └── diffusion/ # 扩散模型管道

│ │

│ ├── distributed/ # 分布式系统

│ │ ├── executor/ # Worker管理、集群、模型更新组

│ │ ├── scheduler/ # 资源管理、生成/奖励调度

│ │ └── strategy/ # 多后端支持抽象层

│ │

│ ├── models/ # 模型支持

│ │ ├── model_provider/ # 不同框架的模型提供者

│ │ └── function_provider/ # 专用操作函数

│ │

│ ├── datasets/ # 数据集加载和处理

│ ├── configs/ # 配置定义

│ ├── utils/ # 工具函数

│ │ ├── collective/ # 分布式训练原语

│ │ ├── checkpoint/ # 检查点管理

│ │ └── metrics/ # 性能监控

│ ├── platforms/ # 平台适配

│ └── third_party/ # 第三方集成

│ ├── vllm/ # vLLM补丁

│ ├── sglang/ # SGLang补丁

│ ├── deepspeed/ # DeepSpeed补丁

│ └── megatron/ # Megatron补丁

│

├── examples/ # 示例配置

│ ├── qwen2.5-*B-*/ # Qwen2.5系列配置

│ ├── qwen3-*B-*/ # Qwen3系列配置

│ ├── start_rlvr_pipeline.py # RLVR启动脚本

│ ├── start_agentic_pipeline.py # Agentic启动脚本

│ └── start_*_pipeline.py # 其他pipeline启动脚本

│

├── tests/ # 测试套件

│ ├── unit/ # 单元测试

│ └── integration/ # 集成测试

│

├── docs/ & docs_roll/ # 文档（中英文）

├── scripts/ # 辅助脚本

├── mcore_adapter/ # Megatron-Core适配器

├── data/ # 数据目录

├── docker/ # Docker配置

└── requirements_*.txt # 依赖文件（不同后端）```
```


  

关键特性

• 配置驱动: 使用Hydra进行层次化YAML配置，支持CLI覆盖

• 多模型支持: Qwen系列（2.5/3/VL），支持0.5B到235B参数规模

• 极致优化: Offload/Reload能力、GPU时分复用、动态设备映射

• 生产就绪: 已在阿里巴巴多个业务场景部署，支持数亿用户规模

