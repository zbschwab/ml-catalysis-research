## Conda Environment Setup

This project uses one primary Conda environment for the notebook and several model-specific environments for ML models with incompatible dependencies.

### 1. Create the primary environment

From the repository root:

```bash
conda env create -f model-scripts/conda_envs/mlip.yml
```

### 2. Create the model environments
(Note: you only have to do this for models you want to rerun!
This is not necessary to play around with existing model output.)

From the repository root (for each model you want to run):
```bash
conda env create -f model-scripts/conda_envs/mlip-mace.yml
conda env create -f model-scripts/conda_envs/mlip-mattersim.yml
conda env create -f model-scripts/conda_envs/mlip-uma.yml
```

### 3. Verify installation

```bash
conda env list
```

You should see:

```text
mlip
mlip-mace
mlip-mattersim
mlip-uma
```

### 4. Run the notebook

Activate your primary notebook environment:

```bash
conda activate mlip
```

The notebook will automatically invoke model scripts using the appropriate Conda environment.

### Notes

* Do not rename the Conda environments; notebook scripts expect the names defined in the YAML files.
* If environment creation fails, remove the environment and recreate it:

```bash
conda env remove -n <environment_name>
conda env create -f <yaml_file>
```
