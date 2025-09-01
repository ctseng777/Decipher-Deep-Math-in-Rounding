# Decipher Deep Math in Rounding: Linear Probing of Mathematical Understanding in Language Models

## Overview

This repository contains the implementation and experiments for investigating how language models understand and process numerical rounding tasks. Through linear probing techniques, we analyze the internal representations of various model architectures to understand how they encode proximity to multiples of 5 and 10.

## Key Contributions

- **Streaming Linear Probes**: Memory-efficient implementation that processes activations in batches rather than storing entire activation matrices
- **Multi-Architecture Analysis**: Comprehensive evaluation across Transformer-based models (Qwen, Dream) and State Space Models (Mamba)
- **Robustness Testing**: Evaluation on both digit and word representations of numbers with template variations
- **Layer-wise Analysis**: Identification of which layers in different architectures best encode numerical proximity information

## Models Evaluated

### Transformer Models

- **Qwen3-4B** ([Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B))
  - **Thinking variant:** [Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)
  - **Non-thinking variant**: [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- **Dream Model**
  - **Dream-7B**: [Dream-org/Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B)

### State Space Models

- **Mamba-1.4B** ([state-spaces/mamba-1.4b-hf](https://huggingface.co/state-spaces/mamba-1.4b-hf))
- **Mamba-2.8B** ([state-spaces/mamba-2.8b-hf](https://huggingface.co/state-spaces/mamba-2.8b-hf))

## Tasks

### Near-5 Classification

Determines if a number is within distance 1 of a multiple of 5 (i.e., last digit in {4, 5, 6}).

### Near-10 Classification

Determines if a number is within distance 1 of a multiple of 10 (i.e., last digit in {0, 1, 9}).

## Technical Approach

### Linear Probing Pipeline

1. **Feature Extraction**: Extract hidden states from each layer of the model
2. **Span Pooling**: Average representations over multi-token numbers
3. **Two-Pass Training**:
   - Pass 1: Fit StandardScalers for normalization
   - Pass 2: Train SGD logistic classifiers
4. **Streaming Processing**: Process data in small batches to minimize memory usage

### Robustness Evaluation

- **Template Variations**: Test on different sentence templates to avoid overfitting
- **Surface Forms**: Evaluate on both digit ("123") and word ("one hundred twenty three") representations
- **Cross-validation**: Train on one template set, evaluate on paraphrased templates

## Repository Structure

```
.
├── Qwen3_4B_thinking_linear_probe_near_5/    # Qwen thinking model, near-5 task
├── Qwen3_4B_thinking_near_10/                # Qwen thinking model, near-10 task
├── Qwen3_4B_non_thinking_linear_probe_near_5/# Qwen non-thinking, near-5 task
├── Qwen3_4B_non_thinking_linear_probe_near_10/# Qwen non-thinking, near-10 task
├── mamba_1.4b_hf_near_5/                     # Mamba 1.4B, near-5 task
├── mamba_1.4b_hf_near_10/                    # Mamba 1.4B, near-10 task
├── mamba_2.8b_hf_near_5/                     # Mamba 2.8B, near-5 task
├── mamba_2.8b_hf_near_10/                    # Mamba 2.8B, near-10 task
├── Dream_linear_probe_near_5/                # Dream model, near-5 task
├── Dream_linear_probe_near_10/               # Dream model, near-10 task
└── DeepMath.pdf                              # Research paper
```

Each directory contains:

- `linear_probe_*.py`: Main probing implementation
- `*_probe_results.json`: Experimental results including layer-wise accuracies

## Key Findings

1. **Layer Specialization**: Different layers show varying levels of numerical understanding, with middle layers often performing best
2. **Architecture Differences**: State Space Models (Mamba) show different patterns compared to Transformer and Diffusion models
3. **Robustness**: Models trained on digit representations generalize reasonably to word representations
4. **Thinking vs Non-Thinking**: "Thinking" model variants show different internal representations compared to standard variants

## Usage

Run a linear probing experiment:

```python
# Example: Run Mamba 1.4B near-5 probing
cd mamba_1.4b_hf_near_5/
python linear_probe_mamba_1.4b_hf_near_5.py
```

Results will be saved as JSON files containing:

- Layer-wise probe accuracies
- Error breakdowns by distance and rounding direction
- Best performing layer identification

## Configuration

Key parameters in each script:

- `TRAIN_N`: Number of training examples (default: 4000)
- `VAL_N`: Number of validation examples (default: 1500)
- `BATCH_SIZE`: Batch size for streaming (adjust based on GPU memory)
- `MAX_LEN`: Maximum sequence length
- `LAYER_STRIDE`: Skip layers for faster probing (1 = probe all)
- `NEAR_THRESHOLD`: Distance threshold for classification (1 for near-5/10)

## Related Work

### Core Methodology

#### Approximation in Empirical Science

- **Generalizing Empirical Adequacy I: Multiplicity and Approximation** – Sebastian Lutz – Proposes a broadened concept of empirical adequacy within constructive empiricism, emphasizing multiplicity and approximation in theoretical–observational relations. [Synthese (2014)](https://link.springer.com/article/10.1007/s11229-014-0440-3)

- **Scientific Hypothesis Generation by Large Language Models: Laboratory Validation in Breast Cancer Treatment** – A. Abdel-Rehim, H. Zenil, O. Orhobor, M. Fisher, R. J. Collins, E. Bourne, G. W. Fearnley, E. Tate, H. X. Smith, L. N. Soldatova, and R. D. King – Demonstrates LLMs generating novel, experimentally validated drug combination hypotheses for breast cancer treatment. [Journal of the Royal Society Interface (2025)](https://royalsocietypublishing.org/doi/10.1098/rsif.2024.0674)

#### Psychology Study on Approximation

- **Children's Number Line Estimation Strategies** – M. Li, J. Yang, and X. Ye – Explores how children use different number line strategies in bounded and unbounded tasks, showing developmental shifts in reference-point use. [Frontiers in Psychology (2024)](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2024.1421821/full)

### Linear Probe

- **Understanding Intermediate Layers Using Linear Classifier Probes** – G. Alain & Y. Bengio – Introduces 'probe' classifiers to examine feature separability and diagnostic behavior across neural network layers. [arXiv (2016)](https://arxiv.org/abs/1610.01644)

### Models

- **Qwen3 Technical Report** – Qwen Team – Presents the Qwen3 family of LLMs featuring dense and Mixture-of-Expert (MoE) models with integrated "thinking" and "non-thinking" modes. [arXiv (2025)](https://arxiv.org/abs/2505.09388)

- **Dream 7B: Diffusion Large Language Models** – J. Ye, Z. Xie, L. Zheng, J. Gao, Z. Wu, X. Jiang, Z. Li, and L. Kong – Introduces Dream 7B, a diffusion-based LLM with iterative denoising generation, excelling in math, coding, and planning tasks. [arXiv (2025)](https://arxiv.org/abs/2508.15487)

- **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** – A. Gu & T. Dao – Proposes the Mamba architecture, a state-space model that scales linearly and outperforms transformers in long-sequence modeling. [arXiv (2023)](https://arxiv.org/abs/2312.00752)

### Model's Numerical Encoding Behaviors

- **Language Models Encode Numbers Using Digit Representations in Base 10** – A. A. Levy & M. Geva – Reveals that LLMs encode numbers via digit-wise base-10 representations, explaining systematic numeric errors. [ACL Anthology (2025)](https://aclanthology.org/2025.naacl-short.33/)

- **Language Model Probabilities are Not Calibrated in Numeric Contexts** – C. Lovering, M. Krumdick, V. D. Lai, V. Reddy, S. Ebner, N. Kumar, R. Koncel-Kedziorski, and C. Tanner – Examines how LMs fail to calibrate probabilities in numeric contexts, even in simple reasoning tasks. [ACL Anthology (2025)](https://aclanthology.org/2025.acl-long.1417/)

### Early Stopping Opportunities

- **BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks** – S. Teerapittayanon, B. McDanel, and H. T. Kung – Proposes BranchyNet, enabling faster inference by allowing early exits from intermediate layers of neural networks. [arXiv (2017)](https://arxiv.org/abs/1709.01686)


## Contact

For questions or collaborations, please open an issue on GitHub or contact the authors through the paper link above.
