# Copilot Instructions for Diffusion-Based Job Scheduling

## Architecture Overview

This project applies diffusion models to solve **Flexible Job Shop Scheduling** (FJSS) problems. The system generates optimal operation-to-machine assignments using a two-stage diffusion pipeline:

1. **Feature Encoding** (`model_enc`): Encodes scheduling features (processing times, job IDs, operation positions) into embeddings
2. **Assignment Generation** (`model_adj`): Denoises random assignments conditioned on encoded features, producing a probability distribution over valid assignments

### Key Components

- **`diffusion.py`**: Two training functions (`feature_diffusion`, `adj_diffusion`) for the two-stage pipeline
- **`inference.py`**: Implements reverse diffusion (denoising loop) to generate assignments from random initialization
- **`utils.py`**: Dataset classes (`Dataset_RL4CO`), instance generation, and scheduling utilities
- **`target_maker.py`**: Generates optimal target solutions using RL4CO models (L2D policy)
- **Models stored in `models/`**: Multiple `.pth` files with different configurations (learning rates: 1e-6, 1e-5, 0.0001, 0.001, 0.01)

## Data Representation

The project encodes scheduling problems as **image-like matrices** (b, c, h, w) for UNet processing:

- **Problem features** → adjacency matrices (b, 1, 20, 20) with:
  - Processing times normalized to [0, 1]
  - Job IDs, operation positions
  - Zero-padded to (20, 20) for variable instance sizes
- **Assignments** → matrices (b, 1, 20, 20) where each cell [op, machine] ∈ {0, 1}
- **Timestep info** → passed as conditioning to DiffUNet via `type_t="timestep"`

Key normalization function: `expand_matrix()` handles padding and min-max normalization.

## Training Workflow

### Model Training
1. **Feature encoder** trained with MSE loss on feature reconstruction
2. **Assignment model** trained to denoise assignments conditioned on encoded features
3. Both use standard diffusion schedule: `β_start ≈ lr`, `β_end = 0.02`, 1000 timesteps
4. Loss functions use torch `MSELoss()(prediction, sampled_noise)`

### Dataset Generation
- `make_dataset(n)`: Creates n FJSS instances with optimal solutions
- Instance generation via `FJSPEnv` from RL4CO with fixed params: 4 jobs, 4 machines, 4 ops/job
- Optimal targets computed using L2D model (`rl4co_model_0.0001.ckpt`)
- Stored as PyTorch `.pt` files in `data/with_targets/`

## Inference

**Reverse diffusion process** in `adj_inference()`:
- Initialize random allocations (b, 1, 20, 20)
- Loop t from timesteps-1 down to 0
- Encode features, concatenate with noisy allocations, predict noise
- Denoise using diffusion formula (formula is standard DDPM)
- Optional: use `torch.topk()` to extract discrete assignments from soft outputs

## HPC Execution

Jobs submitted via SLURM scripts in `job/`:
- **`job_hpc6.sh`**, **`job_hpc7.sh`**, **`job_hpc8.sh`**: GPU jobs with 2 GPUs per node
- Uses custom environment: `/cluster/datastore/vemundvb/enviroments/diff_env/`
- Entry point: `python -u main.py` (note: `-u` for unbuffered output)

**Key dependencies** from `req.txt`:
- PyTorch Lightning 2.5.5, PyTorch 2.8.0
- RL4CO 0.6.0 (FJSS environment & L2D model)
- DeepInv 0.3.4 (DiffUNet architecture)
- TorchRL 0.10.0

## Project-Specific Patterns

### Logging
- **Consistent pattern**: `import logging` with basicConfig at module start
- **Format**: `"%(filename)s:%(lineno)d - %(message)s"`
- Used throughout for debugging shapes, loss values, and progress

### Batch Shape Handling
- Features are multi-channel: `features = torch.cat([proc_times, job_id, pos_job], dim=1)`
- Always expand dims for broadcasting with timesteps: `t[:, None, None, None]`
- Watch for shape mismatches between `noised` and `model_input` concatenations

### Model Checkpointing
- Save models after each epoch: `torch.save(model.state_dict(), path)`
- Models stored with descriptive names: `adj_{grid_size}_{learning_rate}.pth`
- Lightning logs in `lightning_logs/versionX/` for trainer-based experiments

### Dataset Classes
- **`Dataset_RL4CO`**: Returns `(target_assignments, proc_times, job_id, pos_job)` tuples
- Called with dataset path, `ordered` flag (for scheduling type), and generator params
- Handles normalization and padding internally

## Common Tasks

### Adding new feature channel
1. Create adjacency matrix via `expand_matrix(tensor, (20,20), (min_val, max_val))`
2. Concatenate in `features = torch.cat([...], dim=1)` (increases input channels)
3. Update model `in_channels` parameter in DiffUNet initialization

### Modifying diffusion schedule
- Edit `beta_start`, `beta_end`, or `timesteps` in training/inference functions
- Recompute alphas schedule: `betas = torch.linspace(beta_start, beta_end, timesteps)`

### Swapping diffusion models
- Change `model_path_enc`, `model_path_adj` arguments in function calls
- Models must have matching `out_channels` with concatenation inputs

### Testing new instance sizes
- Generator params control problem: `num_jobs`, `num_machines`, `min_ops_per_job`
- Update padding size in `expand_matrix()` calls if not 20×20

## Known Issues & TODOs

- Incomplete diffusion function in `diffusion.py` (line ~370: `return` statement halts training)
- Model saving syntax appears incorrect in some places (should be `torch.save(state_dict, path)`, not model object)
- Feature vector dimensionality increases quickly with concatenation—monitor memory usage
- Inference code has duplicate function definitions (multiple `schedule_actions`, `make_target`)
