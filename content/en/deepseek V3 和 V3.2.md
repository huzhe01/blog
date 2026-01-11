# DeepSeek V3 & V3.2 Technical Deep Dive: Anatomy of a GPT Model

This report is based on the source code analysis of `DeepSeek-V3` and `DeepSeek-V3.2-Exp`, providing a detailed interpretation of the internal structure of a modern GPT (Transformer) large model.

## 1. The Skeleton of the GPT Model

In `inference/model.py`, the `Transformer` class defines the entire model. A standard GPT model is like a multi-layered burger: data (Tokens) enters from the bottom, passes through layers of processing, and finally outputs prediction results from the top.

### Core Components

1.  **Embedding (ParallelEmbedding)**:
    Converts input number IDs (e.g., `12345`) into a high-dimensional vector (Dimension `7168`). This is the first step for the model to "understand" word meanings.
2.  **Transformer Block (Block)**:
    The body of the model, stacked with dozens of layers (61 layers for V3/V3.2). Each layer contains two core parts:
    -   **Attention**: Allows the model to see context and understand relationships between words.
    -   **FFN (Feed-Forward Network)**: The "memory" and "reasoning" center of the model. V3/V3.2 uses a mix of MLP and MoE.
3.  **Output Head**:
    Converts the vector from the last layer back to the vocabulary size (`129280`) to calculate the probability of the next word.

## 2. Core Organ: Attention Mechanism (MLA)

DeepSeek uses **MLA (Multi-Head Latent Attention)**, an optimized attention mechanism.

### 2.1 How MLA Works

Standard attention (MHA) requires storing a huge KV Cache (Key-Value Cache). MLA significantly reduces VRAM usage through **Low-Rank Compression**.

-   **Compression**:
    The input vector first goes through a "compression layer" (`wkv_a`), reducing the dimension to a very low value (`kv_lora_rank=512`). This is the "Latent" state.
-   **Decompression/Projection**:
    When calculating attention, it is projected back to multi-head form via `wkv_b`.

### 2.2 Evolution in V3.2: Sparse Attention with Indexer

V3 uses "Dense Attention," meaning every word looks at all preceding words. V3.2 introduces the **Indexer** module to achieve "Sparse Attention."

-   **Indexer**: A lightweight predictor. It pre-determines: "In this haystack, which 2048 (`topk`) historical words are most important for the current one?"
-   **Hadamard Transform**: V3.2 uses `rotate_activation` (Hadamard Transform) to process Key/Query. This mathematical trick makes vector distribution more uniform, facilitating subsequent FP8 quantization.
-   **Focus on the Important**: Only the Top-K words selected by the Indexer participate in the final attention calculation. This makes the model extremely fast when processing ultra-long texts.

## 3. Core Organ: Mixture of Experts (MoE)

This is the key to DeepSeek's power. Instead of having one giant brain process all information, it has many "Experts."

### 3.1 Gate

-   **Gate**: Acts like a triage desk. When a word comes in, the Gate calculates which domain it belongs to.
-   **Distribution**:
    -   `n_routed_experts = 256`: Total of 256 experts.
    -   `n_activated_experts = 8`: For each word, only the 8 most knowledgeable experts are dispatched to handle it.
    -   `shared_experts`: There is also 1 "Shared Expert" (General Practitioner) who looks at every word to ensure basic capabilities.

### 3.2 Expert

Each expert is essentially a small **MLP** (Multi-Layer Perceptron).

-   Code Structure: `Up_Proj` -> `SiLU` -> `Down_Proj`.
-   They learn different knowledge patterns; some understand code, some math, some literature.

## 4. Neural Conduction: RMSNorm

To ensure signals transmit stably through deep networks without "gradient explosion" or "vanishing," normalization is applied between layers.

-   **V3**: Standard `RMSNorm`.
-   **V3.2**: **Fused RMSNorm**. It combines the `Add` (Residual Connection) and `Norm` (Normalization) operations, reducing memory read/write times and boosting inference speed.

## 5. The Numbers (Parameter Audit)

Actual calculation results based on DeepSeek-V3 671B configuration:

### 5.1 Component-Level Parameters

| Component | Count | Description |
| :--- | :--- | :--- |
| **Embedding** | **927M** | Vocab size 129280 × Dim 7168 |
| **MLA Attention** | **187M / layer** | Includes compression & decompression matrices, extremely lightweight |
| **Dense MLP** | **396M / layer** | Used in the first 3 layers (Dense Block) |
| **MoE Layer** | **11.32B / layer** | Total parameters of all 256 experts |
| **Output Head** | **927M** | Shared with Embedding or independent (calculated as independent here) |
| **Indexer (V3.2)** | **14M** | Sparse attention achieved at negligible cost |

### 5.2 Layer Structure Comparison

DeepSeek-V3 has 61 layers in total:

| Layer Type | Range | Composition | Total Params / Layer | Active Params / Layer (Inference) |
| :--- | :--- | :--- | :--- | :--- |
| **Dense Block** | Layers 0 - 2 (3 layers) | MLA + MLP | ~580M | ~580M |
| **MoE Block** | Layers 3 - 60 (58 layers) | MLA + MoE | **~11.51B** | **~590M** |

> **💡 Key Insight**: Although the MoE layer has a total parameter count of **11.5 Billion**, only a fraction of experts are activated via the Gate mechanism during inference. The actual computation is equivalent to a **590 Million** parameter model! This is why DeepSeek-V3 has a massive capacity of 671B but runs as fast as a 37B model.

## 6. Deep Dive: MoE vs. Dense MLP

The core highlight of DeepSeek-V3 is upgrading the traditional Dense MLP to the **DeepSeekMoE** structure. Let's look at the code to see the differences.

### 6.1 Traditional Heart: Dense MLP

This is the most basic structure, used in the first 3 layers of V3. It's a simple, brutish "Big Guy."

-   **Code Structure**:
    ```python
    class MLP(nn.Module):
        def forward(self, x):
            # SwiGLU Activation: (x * W1) * SiLU(x * W3) * W2
            return self.w2(F.silu(self.w1(x)) * self.w3(x))
    ```
-   **Characteristic**: **All Hands on Deck**. No matter what the input is, all parameters (W1, W2, W3) must participate in the calculation.
-   **Diagram**:
    ![](https://cdn.jsdelivr.net/gh/huzhe01/picsave/mypic20260112010251809.png)

### 6.2 Smart Heart: DeepSeekMoE

This is the main workhorse of V3/V3.2 (last 58 layers). It splits the "Big Guy" into countless "Small Sprites" and introduces a **Shared Expert** mechanism.

-   **Formula**: $$ Output = \text{SharedExpert}(x) + \sum_{i \in TopK} Gate(x)_i \cdot \text{RoutedExpert}_i(x) $$
-   **Core Components**:
    1.  **Shared Expert**:
        -   Also an MLP, but "public."
        -   **Purpose**: Captures general knowledge (like grammatical structure), ensuring a solid foundation regardless of which expert is chosen.
    2.  **Gate**:
        -   A linear classifier. It scores the 256 experts and selects the top 8 (`topk`).
    3.  **Routed Experts**:
        -   256 small MLPs.
        -   **Characteristic**: **Conditional Computation**. Only the 8 experts selected by the Gate actually run; the others "sleep."
-   **Diagram**:
    ![](https://cdn.jsdelivr.net/gh/huzhe01/picsave/mypic20260112010721657.png)

### 6.3 Key Differences Table

| Feature | Dense MLP | DeepSeekMoE |
| :--- | :--- | :--- |
| **Metaphor** | **General Practitioner**: Knows a bit of everything, uses all brain cells for one patient. | **Specialist Consultation**: 1 GP + 256 Specialists. GP sees patient first, then refers to the 8 most relevant specialists. |
| **Parameter Activation** | 100% Activated | **~5% Activated** (8/256 + Shared) |
| **Compute Cost** | High (Linear with parameter size) | **Low** (Huge parameters, small compute) |
| **Specialization** | Knowledge mixed in same weights, prone to "catastrophic forgetting" | Different experts focus on different domains (Code, Math, Lit), decoupling knowledge |
| **Implementation** | Simple Matrix Multiplication | Requires `Gate` scoring, `TopK` selection, then sparse calculation (`gather` + `gemm` + `scatter`) |

## 7. Summary: A "Portrait" of a GPT Model

Imagine a DeepSeek V3/V3.2 model thinking:

1.  **Input**: You ask "What is its principle?"
2.  **Layer 1 (MLA)**: Attention is activated, "its" refers to "GPT model" in the context.
    -   *V3.2 exclusive*: Indexer quickly scans thousands of preceding words, locking onto keywords like "DeepSeek", "Architecture".
3.  **Routing (MoE Gate)**:
    -   Gate judges this as a "Tech/Explanation" question.
    -   Activates 8 neural modules including "Tech Explanation Expert" and "Logical Reasoning Expert".
4.  **Progression**: Signal passes through 61 such layers, information being constantly abstracted and refined.
5.  **Output**: The final layer calculates the most probable next character — "is".

This is the technical detail of GPT presented within the DeepSeek codebase.

## 8. Deep Q&A: The Soul and Evolution of Transformer

### 8.1 Why Attention + MLP?

Each layer of Transformer can be seen as alternating between **"Information Exchange" (Attention)** and **"Information Digestion" (MLP)**.

#### **1. Attention: The "Router" of Information**

-   **Role**: **"Look Around"**.
    -   Before the Attention layer, each Token (e.g., "Apple") is an isolated vector.
    -   Attention allows "Apple" to see "Phone" or "Fruit" in the context.
    -   **Essence**: Weighted Average. It "moves" information from other tokens in the context by calculating relevance.
    -   **Why it works**: Despite various optimizations (MLA, Sparse Attention), its core math $Softmax(QK^T)V$ mimics human "focus" — filtering key parts from messy information.

#### **2. MLP (Multi-Layer Perceptron): The "Processing Plant"**

-   **Role**: **"Think Self"**.
    -   Attention brought the information, but the Token itself hasn't "understood" it yet.
    -   MLP is a non-linear transformation capable of fitting any function. It performs complex reasoning on the mixed information collected by Attention.
    -   **Essence**: Key-Value Memory Network. Massive parameters store static knowledge of the world (e.g., "Apple is red", "Apple Phone is tech").

> **Metaphor**:
>
> -   **Attention** is like a conference table where everyone exchanges information.
> -   **MLP** is like everyone returning to their desk to process what they just heard, combining it with their own knowledge (MLP params) to independently think and update their views.

### 8.2 Why does a Dense MLP layer have hundreds of millions of parameters?

Let's look at DeepSeek-V3's math:

-   **Dim ($d$)**: 7168
-   **Intermediate Dim ($d'$)**: 18432
-   **Structure**: SwiGLU needs 3 matrices ($W_{up}, W_{gate}, W_{down}$)
    $$ Params = d \times d' \times 3 \approx 396 \text{ Million} $$

**Why so many?** MLP bears the burden of "Storing Knowledge." You can think of it as the **model's cerebral cortex**. The wider it is (18432 dim), the finer the granularity of concepts it can distinguish; the more parameters, the more knowledge (facts, common sense, logic rules) it can rote memorize. This is also why MoE works: we don't need to active all "brain cells" every time; doing math only activates the hundreds of millions of parameters in the "Math Zone".

### 8.3 GPT(2) vs BERT: Winning on "Direction"

GPT2 and BERT share the same origin but chose different paths.

-   **BERT (Bidirectional)**: **Master of Cloze Tests**.
    -   Task: `[MASK] is the capital of China` -> Predict `Beijing`.
    -   Pros: Sees full context simultaneously, strong understanding.
    -   Cons: **Can't Speak**. It doesn't know how to generate a sentence from left to right, only valid for fill-in-the-blanks. Thus used mostly for discrimination tasks (Classification, Search).
-   **GPT (Unidirectional)**: **Master of Text Completion**.
    -   Task: `Beijing is` -> Predict `China` -> Predict `'s` ...
    -   Pros: **Generation Capability**. It simulates human speech and thought processes (think of the previous word, then say the next).
    -   **Reason for Victory**: As models grew, people found that "Next Token Prediction" forces the model to learn logical reasoning. To predict accurately, it must understand the world.
    -   **Superiority**: **Universality**. GPT architecture can solve translation, Q&A, summarization, etc., in one way (generation), whereas BERT needs custom Heads for tasks.

This is why today, the GPT (Decoder-only) architecture dominates the LLM field.
