Perfect 👍 let me give you a **clean new section** for your README that covers creating the environment **and** installing `llama-cpp-python` with **CUDA 12.4 GPU support**.

Here’s the updated section ⬇️

---

### 1. Create Environment

We recommend using **conda** (or Miniconda) to isolate dependencies.
This example creates a fresh environment with Python 3.11:

```bash
conda create -n rag_env python=3.11 -y
conda activate rag_env
```

---

### 2. Install `llama-cpp-python` with GPU (CUDA 12.4)

To enable GPU acceleration, you need to compile `llama-cpp-python` from source with CUDA flags.

Run the following inside the environment:

```bash
# uninstall if previously installed
pip uninstall -y llama-cpp-python

# set CUDA build flags (new GGML system)
set CMAKE_ARGS=-DGGML_CUDA=on -DGGML_CUDA_F16=on
set FORCE_CMAKE=1

# install with CUDA support
pip install --force-reinstall --upgrade --no-cache-dir llama-cpp-python --verbose
```

---

### 3. Verify GPU Support

Run this test:

```python
from llama_cpp import llama_supports_gpu_offload
print("CUDA enabled:", llama_supports_gpu_offload())
```

Expected output:

```
CUDA enabled: True
```

---

Would you like me to **merge this into your existing README structure** (right before the `.env` section you showed me earlier), so it flows as a step-by-step setup guide?
