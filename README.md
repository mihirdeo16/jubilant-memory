# Jubilant Memory

## Project Overview

The purpose of this repository is to serve as a personal playground for practicing programming and implementing theoretical concepts. It's a space to translate ideas from theory into working code.

## Directory Structure

```
├── deep_learning
│   ├── huggingface_script
│   │   ├── hf_inference.py
│   │   ├── hf_sft_extraction.py
│   │   ├── hf_textprocessing.py
│   │   └── hf_train.py
│   └── models_architectures
│       ├── auto_encoders.py
│       ├── cnn_vision.py
│       └── feedforward_nn.py
├── machine_learning
│   ├── main.py
│   ├── mlp.py
│   ├── nearest_neighbors.py
│   └── tf_idf.py
├── oop_principles
│   └── items.py
└── statistical_analysis
    ├── correlation.py
    ├── means.py
    ├── running_mean.py
    ├── running_variance.py
    └── simulation_discrete.py
```

## Directory Descriptions

*   **`deep_learning`**: This directory contains scripts and implementations related to deep learning concepts, models, and architectures. It includes subdirectories for `huggingface_script` and `models_architectures`.

*   **`machine_learning`**: This directory is for implementing various machine learning algorithms and concepts from scratch.

*   **`oop_principles`**: This directory holds examples and implementations of Object-Oriented Programming (OOP) principles and design patterns.

*   **`statistical_analysis`**: This directory contains scripts for statistical analysis, probability simulations, and related concepts.


### Basic of Data Types & Memory allocation
Most fundamental unit of computer is A **bit** which has values of is zero & one. (0 & 1)
A group of 8 bits is called a **byte**. A byte can represent 256 different values (2^8 = 256).
Here is a table showing common data types and their sizes in bytes and bits:
| Data Type | Size (Bytes) | Size (Bits) |
|-----------|--------------|-------------|
| byte      | 1            | 8           |
| Char      | 1            | 8           |
| Int8      | 1            | 8           |
| Int16     | 2            | 16          |
| Int32     | 4            | 32          |
| float16   | 2            | 16          |
| bfloat16  | 2            | 16          |
| float32   | 4            | 32          |

*Note: **float16** and **bfloat16** are both 16-bit floating-point formats, but they have different representations and ranges. While `float16` is designed for precision after decimal points, `bfloat16` is optimized for a wider range of values, making it more suitable for deep learning applications.*
The two most common quantization cases are float32 -> float16 and float32 -> int8.

Quick reference for C data types:
| Data Type | Size (Bytes) |
|-----------|--------------|
| char      | 1            |
| int       | 2 or 4       |
| float     | 4            |
| double    | 8            |

