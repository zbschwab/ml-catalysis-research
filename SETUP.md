# Setup Guide — MLIP vs DFT Comparison

This guide sets up everything the **`CO3_on_Ag.ipynb`** notebook needs. You only
do this once. When you're done, open the notebook, fill in **Section 0**, and run.

Each of the three models (eSEN, UMA, Orb) lives in its **own conda environment**
because they need conflicting versions of PyTorch and its dependencies. The
notebook runs each model in its own environment automatically — you just have to
create them here first.

All commands go in **Anaconda Prompt** (search "Anaconda Prompt" in the Windows
Start menu). Run them one line at a time.

---

## Step A — Find your conda path

```bash
where conda
```

Copy the line ending in `Scripts\conda.exe` (e.g.
`C:\Users\YourName\anaconda3\Scripts\conda.exe`) and paste it into `CONDA_EXE`
in Section 0 of the notebook.

---

## Step B — Hugging Face account, licenses, and token

The eSEN and UMA model weights are downloaded from Hugging Face and are gated,
so you must accept their licenses once in a browser:

1. Create a free account at <https://huggingface.co>
2. Accept the eSEN/OMAT24 license at <https://huggingface.co/facebook/OMAT24> (click **Agree**)
3. Request access to UMA at <https://huggingface.co/facebook/UMA> (click **Agree**)
4. Create an access token at <https://huggingface.co/settings/tokens>
   (**New token → Read role → Generate**, then copy it)

Paste the token into `HF_TOKEN` in Section 0 of the notebook. Orb does **not**
need a token — it downloads its weights automatically.

---

## Step C — Decide: CPU or GPU?

Everything works on CPU. A GPU makes the self-relaxation step much faster but is
optional. How well a GPU is supported depends entirely on what PyTorch and these
libraries (fairchem, orb-models) support — not on this notebook. Here is the
honest support picture, best to worst:

| GPU vendor | Backend | Support level | Notebook auto-detects? |
|------------|---------|---------------|------------------------|
| **NVIDIA** | CUDA | Full, all platforms, all three models | Yes |
| **AMD** (Linux) | ROCm | Good on Linux; reuses the CUDA code path | Yes (see AMD section) |
| **AMD** (Windows) | DirectML | Experimental, limited | No (manual) |
| **Apple Silicon** | MPS | Orb only; eSEN/UMA stay on CPU | Yes (Orb only) |
| **Intel Arc / Data Center** | XPU | Works for UMA and Orb; eSEN stays on CPU | Yes (UMA & Orb) |

The two common cases — NVIDIA, and "no GPU / Mac" — are covered in the main
steps below. AMD, Apple, and Intel have their own short sections after Step D.

**To check if you have a usable NVIDIA GPU**, run:

```bash
nvidia-smi
```

- If you see a table with your GPU name and a **`CUDA Version: XX.X`** in the
  top-right corner → you can use the **GPU path** below. Note that CUDA number.
- If the command is "not recognized" or shows no GPU → use the **CPU-only path**.
  (Apple Silicon Macs: use CPU-only here; Orb will still auto-use the Mac GPU.)

> **Key idea about the CUDA number:** the version from `nvidia-smi` is the
> *highest* your driver supports — you do **not** install that exact version,
> and you do **not** need the CUDA Toolkit. PyTorch's pip packages bundle their
> own CUDA runtime. You just pick a PyTorch CUDA build whose number is **≤** the
> number `nvidia-smi` shows. NVIDIA drivers are backward compatible, so a 13.2
> driver happily runs a `cu130`, `cu129`, or `cu126` PyTorch build.

### Picking your CUDA build tag

PyTorch publishes builds for a handful of CUDA versions (the "cuXXX" tag). Pick
the **highest tag that is ≤ your driver's CUDA version**. Common tags:

| Your `nvidia-smi` CUDA version | Use this PyTorch tag |
|--------------------------------|----------------------|
| 13.0 or higher                 | `cu130`              |
| 12.9 – 12.x                    | `cu129`              |
| 12.6 – 12.8                    | `cu126`              |
| 11.8 – 12.5                    | `cu118`              |

If a specific torch version doesn't offer your first-choice tag, step down to the
next lower one — it will still run. The current list of tags for any version is
at <https://pytorch.org/get-started/locally/> and the previous-version commands
are at <https://pytorch.org/get-started/previous-versions/>.

In the commands below, **replace `cuXXX` with the tag you picked.**

---

## Step D — Create the three environments

Note: All enviroments are DEFAULTED to {ModelName}_CPU, this does not mean they solely run off of the CPU. 
The names are able to be changed, just MAKE SURE to change them in Section 0 of the notebook.

Pick **one** column: follow the GPU commands *or* the CPU-only commands for each
environment. Don't mix them within a single environment.

### eSEN environment (fairchem **v1** — the delicate one)

eSEN-30M-OAM is a fairchem **version 1** model. It needs the older fairchem, a
specific scipy, and PyTorch Geometric extensions (`torch_scatter` etc.) whose
wheels must match the exact torch version **and** CUDA tag. We pin torch to
`2.4.1` because that's what fairchem v1 and its PyG wheels are built for.

**GPU:**
```bash
conda create -n eSEN_CPU_2 python=3.11 -y
conda activate eSEN_CPU_2
pip install fairchem-core==1.10.0
pip install "scipy<1.15"
pip install ase huggingface_hub ipykernel
pip install "torch==2.4.1" --index-url https://download.pytorch.org/whl/cuXXX
pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.4.1+cuXXX.html
python -m ipykernel install --user --name eSEN_CPU_2 --display-name "Python (eSEN_CPU_2)"
```
> torch 2.4.1 offers `cu118`, `cu121`, and `cu124`. For most modern drivers use
> `cu124`. Use the **same** tag in both the torch line and the PyG wheel line.

**CPU-only:**
```bash
conda create -n eSEN_CPU_2 python=3.11 -y
conda activate eSEN_CPU_2
pip install fairchem-core==1.10.0
pip install "scipy<1.15"
pip install ase huggingface_hub ipykernel
pip install "torch==2.4.1" --index-url https://download.pytorch.org/whl/cpu
pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.4.1+cpu.html
python -m ipykernel install --user --name eSEN_CPU_2 --display-name "Python (eSEN_CPU_2)"
```

### UMA environment (fairchem **v2** — current)

fairchem v2 pins a specific torch (currently the 2.8 series). Install fairchem
first, then **let it tell you** which torch it wants, and install the matching
CUDA build of that exact version.

**GPU:**
```bash
conda create -n UMA_CPU python=3.11 -y
conda activate UMA_CPU
pip install fairchem-core
pip install ase ipykernel
python -c "import torch; print(torch.__version__)"
```
The last line prints something like `2.8.0+cpu`. Take the version number before
the `+` (e.g. `2.8.0`) and install the CUDA build of that version:
```bash
pip uninstall torch -y
pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cuXXX
python -m ipykernel install --user --name UMA_CPU --display-name "Python (UMA_CPU)"
```
> Swap `2.8.0` for whatever the version check printed, and `cuXXX` for your tag.

**CPU-only:**
```bash
conda create -n UMA_CPU python=3.11 -y
conda activate UMA_CPU
pip install fairchem-core
pip install ase ipykernel
python -m ipykernel install --user --name UMA_CPU --display-name "Python (UMA_CPU)"
```
(fairchem installs a working CPU torch by default, so nothing extra is needed.)

### Orb environment

Orb is the most forgiving — no pinned PyG wheels. Install orb-models, check the
torch version it brought in, and (for GPU) swap in the matching CUDA build.

**GPU:**
```bash
conda create -n Orb_CPU python=3.11 -y
conda activate Orb_CPU
pip install orb-models
pip install ase ipykernel
python -c "import torch; print(torch.__version__)"
```
Take the printed version (e.g. `2.12.0`) and install its CUDA build:
```bash
pip uninstall torch -y
pip install "torch==2.12.0" --index-url https://download.pytorch.org/whl/cuXXX
python -m ipykernel install --user --name Orb_CPU --display-name "Python (Orb_CPU)"
```
> Swap `2.12.0` for whatever the version check printed, and `cuXXX` for your tag.
> Recent torch versions default to a CUDA build already — if the version check
> shows `+cuXXX` (not `+cpu`), it's already GPU-ready and you can skip the swap.

**CPU-only:**
```bash
conda create -n Orb_CPU python=3.11 -y
conda activate Orb_CPU
pip install orb-models
pip install ase ipykernel
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m ipykernel install --user --name Orb_CPU --display-name "Python (Orb_CPU)"
```

---

## Step D2 — Non-NVIDIA GPUs (AMD, Apple, Intel)

These are alternatives to the CUDA install in Step D. Use the relevant section
*instead of* the GPU torch line for each environment; everything else in Step D
(fairchem/orb-models, scipy, ase, ipykernel, the kernel registration) stays the
same.

### AMD GPUs on Linux (ROCm)

ROCm is AMD's CUDA-equivalent. On **Linux**, PyTorch ships ROCm wheels, and
helpfully PyTorch reuses the CUDA API for them — so `torch.cuda.is_available()`
returns `True` on a working ROCm setup, and **this notebook's `USE_GPU` detection
works for AMD on Linux with no code changes.**

Install the ROCm build instead of the CUDA build, picking the ROCm version that
matches your driver (check the current tag at
<https://pytorch.org/get-started/locally/> — e.g. `rocm6.2`):

```bash
# UMA / Orb — replace the CUDA torch line with (example for ROCm 6.2):
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
```

Verify the same way as NVIDIA — `torch.cuda.is_available()` should print `True`.

Notes and caveats:
- **eSEN is unlikely to work on AMD**: its pinned `torch_scatter`/`torch_sparse`
  wheels are published for CPU and CUDA only, not ROCm. Run eSEN on CPU and use
  ROCm for UMA and Orb.
- Only certain AMD cards (mostly Radeon RX / Instinct on supported Linux distros)
  are supported. Check AMD's compatibility list:
  <https://rocm.docs.amd.com/projects/radeon/en/latest/docs/compatibility/native_linux/native_linux_compatibility.html>

### AMD GPUs on Windows (DirectML)

ROCm does not run natively on Windows. The options are: (1) use **WSL2** and
follow the Linux ROCm steps inside it, or (2) use the experimental **DirectML**
backend. DirectML uses a separate device API (`torch_directml.device()`), which
this notebook does not call, so it would need manual code changes and isn't
recommended here. For AMD on Windows, CPU-only is the reliable path unless you
set up WSL2.

### Apple Silicon (M1/M2/M3 — MPS)

On a Mac with Apple Silicon, PyTorch's **MPS** backend gives GPU acceleration.
Support here is partial:
- **Orb** can use MPS — the notebook detects it automatically when `USE_GPU` is
  `'auto'` or `True`, and the Orb script already includes an MPS code path.
- **eSEN and UMA** (fairchem) do **not** support MPS, so they stay on CPU on a
  Mac. That's expected, not an error.

No special install is needed — the default `pip install torch` on macOS includes
MPS. Just create the environments with the CPU-only commands from Step D; Orb
will still reach the Mac GPU on its own. (On Intel Macs there is no MPS; use CPU.)

### Intel GPUs (Arc, Data Center Max — XPU)

PyTorch supports Intel GPUs through the **XPU** backend, installed from a
dedicated index. **The notebook now detects Intel XPU automatically** (it checks
`torch.xpu.is_available()` and uses the `"xpu"` device), so UMA and Orb will use
an Intel GPU when `USE_GPU` is `'auto'` or `True`.

Install the XPU build instead of the CUDA build for the UMA and Orb environments:

```bash
# UMA and Orb — instead of the CUDA torch line:
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/xpu
```

Then verify (note XPU uses its own check, not `cuda`):

```bash
python -c "import torch; print(torch.__version__, torch.xpu.is_available())"
```

You want `True` at the end. When you run the notebook, UMA and Orb should print
`Compute device: xpu`.

Requirements and caveats:
- You need the **Intel GPU driver** and (on some setups) the **Intel Deep
  Learning Essentials / oneAPI** runtime installed. See the official guide:
  <https://docs.pytorch.org/docs/main/notes/get_start_xpu.html>
- Supported on **Intel Arc** discrete GPUs, **Intel Core Ultra** with built-in
  Arc graphics, and **Data Center GPU Max**, on Windows 10/11 and Linux.
- **eSEN stays on CPU** on Intel GPUs: fairchem v1 has no XPU path and its
  `torch_scatter`/`torch_sparse` wheels are CPU/CUDA only. The eSEN script
  detects the XPU, prints a note explaining this, and runs on CPU. UMA and Orb
  still use the Intel GPU.

---

## Step E — Verify everything

Confirm the three kernels are registered:

```bash
jupyter kernelspec list
```

You should see `eSEN_CPU_2`, `UMA_CPU`, and `Orb_CPU`.

If you set up for GPU, confirm each environment's torch can see it:

```bash
conda activate eSEN_CPU_2
python -c "import torch; print('eSEN', torch.__version__, torch.cuda.is_available())"
conda activate UMA_CPU
python -c "import torch; print('UMA', torch.__version__, torch.cuda.is_available())"
conda activate Orb_CPU
python -c "import torch; print('Orb', torch.__version__, torch.cuda.is_available())"
```

For GPU you want `True` at the end of each line. For CPU-only, `False` is correct
and expected.

> **Intel XPU users:** the `torch.cuda.is_available()` check above will show
> `False` even when your Intel GPU is working — XPU is a separate backend. Check
> it instead with `python -c "import torch; print(torch.xpu.is_available())"`,
> which should print `True`. (eSEN will still use CPU on Intel; that's expected.)

For eSEN specifically, also confirm the PyG extensions import (this catches a
torch/wheel mismatch):

```bash
conda activate eSEN_CPU_2
python -c "import torch_scatter, torch_sparse, torch_cluster; print('PyG extensions OK')"
```

---

## Step F — Configure and run the notebook

1. Open `CO3_on_Ag.ipynb`.
2. In **Section 0**, set:
   - `CONDA_EXE` to the path from Step A
   - `HF_TOKEN` to the token from Step B
   - the three `*_ENV_NAME` values to match the environment names you created
     (`eSEN_CPU_2`, `UMA_CPU`, `Orb_CPU` if you used the names above)
   - `USE_GPU` to `'auto'` (GPU if available, else CPU), `True` (force GPU), or
     `False` (force CPU)
3. Run the cells top to bottom.

Each model prints `Compute device: ...` when it starts, so you can confirm it
picked up the GPU.

---

## Naming note

The environment names you create **must exactly match** the `*_ENV_NAME`
variables in Section 0. The names above (`eSEN_CPU_2`, `UMA_CPU`, `Orb_CPU`) are
just the defaults — the `_CPU` suffix is only a label and does **not** force CPU
mode. If you rename an environment, update Section 0 to match.

---

## Troubleshooting

The eSEN (fairchem v1) environment is the most likely to need extra care because
it's an older library among newer dependencies. Common issues and fixes:

- **`ImportError: cannot import name 'X' from 'Y'`** — a dependency moved on past
  what fairchem v1 expects. Downgrade that one package, e.g. `pip install "Y<version"`.
  Already handled in Step D: `scipy<1.15` and the pinned PyG wheels.
- **`ModuleNotFoundError: No module named 'torch_scatter'`** — the PyG wheels
  didn't install; re-run the `torch_scatter ...` line, making sure the torch
  version and `cuXXX` tag in the URL exactly match your installed torch.
- **PyG extensions import error about an "undefined symbol" or version** — the
  wheels don't match torch. Run `python -c "import torch; print(torch.__version__)"`
  and reinstall the wheels from the matching `https://data.pyg.org/whl/torch-<that version>.html`.
- **`fairchem-core ... requires torch~=2.8.0, but you have torch X`** — you
  installed a torch version fairchem v2 doesn't accept. Install the CUDA build of
  the exact version it asks for (the `~=2.8.0` means any `2.8.x`).
- **Compiler `cl` is not found (Orb)** — handled in the notebook, which disables
  `torch.compile`. No action needed.

GPU-specific:

- **Notebook says `Compute device: CPU` even though you installed GPU torch** —
  check that environment with `python -c "import torch; print(torch.cuda.is_available())"`.
  If that's `False`, the CPU build is still installed; redo the torch swap with a
  `cuXXX` index URL. If it's `True` but the notebook still says CPU, re-run the
  notebook's Section 1 to regenerate the scripts.
- **GPU types:** any CUDA-capable **NVIDIA** card works everywhere (GTX, RTX,
  Quadro, Tesla, A100, H100, …). **AMD** (ROCm on Linux), **Apple Silicon** (MPS,
  Orb only), and **Intel** (XPU, for UMA and Orb) are covered in Step D2 — read
  that section for the support level and caveats of each. In all non-NVIDIA
  cases, **eSEN stays on CPU** because its PyTorch Geometric wheels are only
  published for CPU and CUDA.
- **Intel GPU not used / `torch.xpu.is_available()` is `False`** — install the
  `xpu` torch build (Step D2 Intel section) and confirm the Intel GPU driver and
  oneAPI runtime are installed. XPU uses `torch.xpu.is_available()`, not the
  `cuda` check.
