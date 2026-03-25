## Custom IsaacLab Diffusion Policy Runner

This folder is a lightweight IsaacLab-side project for running a trained diffusion policy checkpoint in:

`Isaac-Lift-Physics-Select-Cube-Franka-IK-Rel-v0`

It assumes:

- IsaacLab is installed in the Python environment you use to launch Isaac Sim
- this repository is available locally, because the script loads the diffusion policy checkpoint and model code from here
- the IsaacLab Python environment also has the diffusion-policy runtime deps needed for checkpoint loading and inference

Minimum extra packages usually needed:

```bash
pip install dill hydra-core omegaconf einops filelock threadpoolctl --no-
pip install diffusers==0.11.1 huggingface_hub==0.25.2

pip install "numcodecs==0.15.1" "zarr==2.18.7"


```

### Run

```bash
python custom_isaaclab_dp/play_diffusion_policy.py \
  --checkpoint /abs/path/to/checkpoint.ckpt \
  --num_episodes 10 \
  --headless
```

### Notes

- The script reads `n_obs_steps`, observation keys, and model config directly from the checkpoint.
- It uses the EMA policy if the checkpoint contains one.
- It now supports both low-dimensional and image-based IsaacLab diffusion-policy checkpoints.
- For image checkpoints, it auto-enables cameras before IsaacLab launch and requests `rgb_array` rendering when needed.
