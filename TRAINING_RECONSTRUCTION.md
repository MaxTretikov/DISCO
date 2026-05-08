# DISCO Training Code Reconstruction

This note maps the paper training description in `paper.md` onto the released
codebase. The short version: the public repository contains the model,
feature path, checkpoint loading, and inference sampler, but not a complete
training runner, PDB dataset, cropper, or loss implementation.

## What Is Present

Training-relevant model components are mostly intact:

- `disco/model/disco.py`
  - `DISCO.get_pairformer_output(...)` builds the trunk representations and
    accepts the training-time noisy structure and sequence noise inputs through
    `xt_noised_struct`, `sigma`, and `sigma_seq`.
  - `DISCO._update_reps_with_seq_struct_encode(...)` implements the paper's
    CrossModalEncode: it predicts preliminary clean sequence/structure, encodes
    `x_t` and `x_0` sequence/structure, and injects them into single/pair reps.
  - `DISCO.apply_subs_parameterization(...)` implements SUBS masking by
    suppressing the mask token, nucleotide tokens, and optionally UNK.
  - `DISCO.forward(...)` is inference-only; it delegates to the sampler instead
    of computing losses.

- `disco/model/modules/diffusion.py`
  - `JointDiffusionModule.forward(...)` returns `(x_denoised, seq_logits)`.
  - It uses separate atom decoders for structure and sequence, matching the
    paper's "Structure Update" and "Sequence Logits" heads.
  - `DiffusionConditioning` embeds structure and sequence noise levels
    separately when `do_fourier_embed_seq: true`.
  - Note: the paper text says the sequence noise is embedded directly, but the
    released code applies the same `log(noise / sigma_data) / 4` transform to
    the sequence conditioning scalar. In inference that scalar is an EDM-like
    `sigma_seq`, not raw mask fraction `r`.

- `disco/model/modules/head.py`
  - `DistogramHead` exists, but it is not called by any released training loop.

- `disco/data/json_to_feature.py`, `disco/data/task_manager.py`,
  `disco/data/featurizer.py`
  - These implement the released JSON/inference feature path.
  - `TaskManager.mask_ref_information(...)` masks sequence-leaking reference
    position, charge, and mask features for masked protein residues.
  - `Featurizer.get_mask_features(...)` exposes the paper's sequence-loss masks:
    `res_occ_cutoff_mask`, `res_is_resolved_mask`, and `histag_mask`.

- `configs/model/default.yaml`
  - Matches Table S1 architecture values: `c_s=384`, `c_z=128`, 8 Pairformer
    blocks, 24 diffusion transformer blocks, 3 atom encoder/decoder blocks,
    `sigma_data=16.0`, DPLM 650M frozen, and Pairformer dropout `0.25`.

## What Is Missing

The public code does not include:

- A `runner/train.py`, `Trainer`, `training_step`, optimizer setup, EMA, or
  scheduler.
- A PDB/Protenix training dataset reader.
- The AF3/Protenix weighted chain/interface sampling logic.
- The AF3-style cropper:
  - 20 percent contiguous crop
  - 40 percent spatial crop
  - 40 percent spatial interface crop
  - crop size 384
- Training-time random sequence masking from sampled `r`.
- Structure noising from sampled `sigma`.
- Ground-truth labels in the released inference feature path. Inference
  features do not carry `coordinate` or `coordinate_mask`.
- The DISCO-specific losses:
  - weighted aligned MSE
  - smooth LDDT
  - masked diffusion sequence cross entropy
  - distogram loss invocation
  - optimal chain assignment before structure losses

OpenFold provides a reference `distogram_loss`, but the aligned MSE and smooth
LDDT described in the paper need to be implemented or borrowed from an AF3-like
codebase.

## Paper Training Recipe

From `paper.md` Appendix A.5 and Table S1:

- Hardware/run: DDP on 32 L40S GPUs for 160,000 steps.
- Trainable parameters: 235M out of 888M total.
- DPLM 650M frozen.
- Structure encoder initialized from LigandMPNN, then trained end to end.
- Dropout only in Pairformer, probability 0.25.
- Sequence batch size: 32.
- Structure batch size: 96.
- Crop size: 384.
- Gradient clipping: 10.0.
- EMA decay: 0.999.
- Adam:
  - learning rate: `1.8e-4`
  - betas: `(0.9, 0.95)`
  - weight decay: `1e-8`
- LR schedule:
  - 1000 step linear warmup
  - multiply by 0.95 every 50,000 steps
- Loss weights:
  - sequence: `1.0`
  - MSE: `4.0`
  - smooth LDDT: `4.0`
  - distogram: `0.03`

Noising:

- Sample structure noise:

```python
sigma = sigma_data * torch.exp(-1.2 + 1.5 * torch.randn(batch_size))
```

- With `sigma_data = 16`.
- Sample sequence time `r ~ Uniform(0, 1)`.
- Paper uses linear schedule `alpha_r = 1 - r`, so each protein sequence token
  is masked with probability `r`.
- Sequence loss weight for masked diffusion is proportional to `1 / r`.
- The exact training value passed as `t_hat_noise_level_seq` is not explicit in
  the released code. Inference converts a sequence schedule value to an EDM-like
  noise level with `InferenceNoiseScheduler.time_to_noise_lvl(...)` before
  calling the diffusion module.

## Likely Training Step Shape

The released model can be trained by bypassing `DISCO.forward(...)` and calling
the lower-level methods directly:

```python
def disco_training_step(model, feat, labels):
    # labels must come from a real training dataset, not the released inference
    # feature path.
    x0_struct = labels["coordinate"]
    coord_mask = labels["coordinate_mask"].bool()
    true_seq = labels["prot_restype_tokens"]
    valid_seq_mask = (
        feat["res_occ_cutoff_mask"].bool()
        & feat["res_is_resolved_mask"].bool()
        & ~feat["histag_mask"].bool()
    )

    batch = x0_struct.shape[0]
    sigma = model.configs.sigma_data * torch.exp(
        -1.2 + 1.5 * torch.randn(batch, device=x0_struct.device)
    )
    xt_struct = x0_struct + sigma[:, None, None] * torch.randn_like(x0_struct)

    r = torch.rand(batch, device=x0_struct.device).clamp_min(1e-4)
    mask = torch.rand_like(true_seq.float()) < r[:, None]
    xt_seq = true_seq.masked_fill(mask, MASK_TOKEN_IDX)
    feat["masked_prot_restype"] = xt_seq

    # IMPORTANT: reference features for masked residues must also be masked,
    # matching TaskManager.mask_ref_information(...).
    # Open reconstruction choice: paper samples raw sequence time r, while the
    # released inference path feeds an EDM-like sigma_seq into the model.
    # A conservative first implementation should make this transform explicit
    # and ablate raw r vs scheduler.time_to_noise_lvl(r).
    sigma_seq = sequence_time_to_model_conditioning(r, model)

    s_inputs, s, z, s_skip, z_skip, _, encoding_dict = model.get_pairformer_output(
        feat,
        N_cycle=model.N_cycle,
        task_info={},
        sigma_seq=sigma_seq,
        xt_noised_struct=xt_struct,
        sigma=sigma,
        inplace_safe=False,
        chunk_size=None,
    )

    pred_struct, seq_logits_pre = model.diffusion_module(
        x_noisy=xt_struct,
        t_hat_noise_level_struct=sigma,
        t_hat_noise_level_seq=sigma_seq,
        input_feature_dict=feat,
        s_inputs=s_inputs,
        s_trunk=s,
        z_trunk=z,
        s_skip=s_skip,
        z_skip=z_skip,
        encoding_dict=encoding_dict,
    )

    seq_log_probs = model.apply_subs_parameterization(
        seq_logits_pre,
        xt_seq,
        exclude_unk=False,
        enforce_unmask_stay=False,
    )

    dist_logits = model.distogram_head(z)

    # Compute:
    # - sequence CE over masked and valid sequence positions, weighted by 1 / r
    # - optimal-chain-assigned weighted aligned MSE
    # - smooth LDDT
    # - AF2 distogram loss
    # Then combine with weights 1, 4, 4, 0.03.
```

This sketch captures the call graph, not a drop-in implementation. In
particular, `labels["prot_restype_tokens"]`, chain assignment, reference feature
remasking, and all losses still need careful implementation.

## Implementation Plan To Rebuild Training

1. Add a training dataset layer.
   - Start from the Protenix/AF3 processed data mentioned in the paper.
   - Emit both model features and labels: coordinates, coordinate masks,
     residue identities, molecule type masks, atom-to-token maps, and chain
     permutation metadata.

2. Add training-time masking/noising transforms.
   - Randomly sample `r` and protein masks per crop.
   - Replace `masked_prot_restype`.
   - Apply the same anti-leakage reference masking currently implemented in
     `TaskManager.mask_ref_information(...)`, but for stochastic masks.
   - Sample `sigma` and construct `xt_struct`.

3. Add a loss module.
   - Reuse OpenFold's `distogram_loss` where possible.
   - Implement paper weighted rigid alignment and weighted MSE.
   - Implement smooth LDDT with 15 A default radius and 30 A for nucleotide
     atoms.
   - Add sequence masked diffusion CE with the validity masks from
     `Featurizer.get_mask_features(...)`.

4. Add `runner/train.py`.
   - Instantiate the same `DISCO` model as inference.
   - Freeze DPLM weights, keep LigandMPNN structure encoder trainable.
   - Use Adam, LR warmup/decay, gradient clipping, DDP/Fabric, and EMA.
   - Save checkpoints in the same shape expected by
     `InferenceRunner.load_checkpoint(...)`: `{"model": state_dict}` at minimum.

5. Validate on tiny synthetic crops before PDB scale.
   - Verify all tensors route through `get_pairformer_output(...)` and
     `JointDiffusionModule.forward(...)`.
   - Check gradients reach Pairformer, diffusion module, sequence atom decoder,
     structure atom decoder, and structure encoder, but not DPLM.
   - Overfit one or two tiny examples to catch masking and shape errors.

## Current Smoke Path

Preprocess one or more local structures into cropped training examples:

```bash
uv run python -m disco.training.preprocess \
  --input output/pdbs/length_70_sample_0.pdb \
  --output-dir /tmp/disco_train_examples \
  --manifest /tmp/disco_train_manifest.txt \
  --crop-size 384
```

The raw Protenix archive is large enough to use directly. Fully materializing
crop-384 `.pt` examples is not recommended on the current archive volume: a
small sample averaged about 22 MiB per crop, which projects to multiple TiB for
the full train split. Prefer the streaming Protenix dataloader with a bounded
cache:

```bash
CUDA_VISIBLE_DEVICES= uv run python runner/train.py \
  training=streaming_protenix \
  training.dataloader.batch_size=2
```

For a small CPU smoke check:

```bash
CUDA_VISIBLE_DEVICES= uv run python runner/train.py \
  training=streaming_protenix \
  experiment=train_smoke \
  logger=csv \
  training.dataloader.batch_size=2 \
  training.dataloader.max_entries=8 \
  training.dataloader.crop_size=32 \
  training.dataloader.cache_dir=/tmp/disco_stream_train_cache \
  training.dataloader.max_cache_gb=0.05
```

This reads Protenix's filtered `before_2021-09-30_res4.5` index, samples
chain/interface rows with cluster-aware weights, builds paper-style crops on
demand, drops the currently-unused dense `bond_mask`, and keeps only generated
crops in the configured LRU cache.

The older fully materialized path is still available for small subsets:

```bash
uv run python -m disco.training.pdb_dataset \
  --archive-root /mnt/archive/datasets/protenix \
  --source protenix \
  --protenix-index /mnt/archive/datasets/protenix/indices/indices_20260107-20chains_before_2021-09-30_res4.5.csv.gz \
  --output-dir /mnt/archive/datasets/disco_training/processed/examples \
  --manifest /mnt/archive/datasets/disco_training/processed/manifest.jsonl \
  --metadata /mnt/archive/datasets/disco_training/processed/metadata.jsonl \
  --crop-size 384 \
  --samples-per-entry 1
```

With `--protenix-index`, this uses Protenix's filtered
`before_2021-09-30_res4.5` train split, chain/interface rows, row-level
`cluster_id`, and `N_clust` counts from the index itself. It also applies the
paper weights and 20/40/40 contiguous/spatial/interface crops. Without that
index, the processor falls back to discovering structures directly, applying
the 2021-09-30 deposition cutoff, and using identical-sequence clusters from
`pdb_seqres.txt.gz` or a file passed with `--cluster-file`.

Run a one-step CPU smoke check for a materialized manifest with a small PLM and
disabled structure encoder:

```bash
CUDA_VISIBLE_DEVICES= uv run python runner/train.py \
  experiment=train_smoke \
  logger=csv \
  training.dataloader.manifest_path=/mnt/archive/datasets/disco_training/processed/manifest.jsonl \
  training.dataloader.batch_size=2 \
  training.dataloader.weighted_sampling=true
```

The dataloader pads variable atom, token, template, and protein-sequence
dimensions inside each batch. Structure losses use `coordinate_mask`;
sequence losses use `true_prot_restype_mask`; distogram targets are scattered
from representative atoms back onto token indices.

## Main Risk

The biggest reconstruction risk is not the neural network call graph. That is
mostly present. The risk is reproducing the training distribution: AF3/Protenix
filtering, clustering, sampling weights, crop semantics, unresolved atom masks,
chain assignment, and leakage-free sequence masking. Those choices are large
enough that a naive trainer may run while still producing a meaningfully
different model.
