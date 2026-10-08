# Code for Paper "Task-hardware co-optimization for universal low-power programmable photonic neural accelerators"

This compact release documents the baseline and task-hardware co-optimization
protocol used in the manuscript.  It is deliberately restricted to the public
method components: the 32×32 CUMEC device model, a task-agnostic training loop,
and the L1, L2 and smoothness (SM) voltage regularizers.  Raw datasets,
checkpoints, result files, notebook outputs and task-specific production
pipelines are not part of this repository.

## What is included

- `src/photonic_training/device/Chip.py`: public 32×32 CUMEC device model.
- `src/photonic_training/regularizers.py`: explicit L1 and SM penalties and
  AdamW construction for L2 weight decay.
- `src/photonic_training/trainer.py`: shared one-epoch training loop.
- `configs/`: four transparent experiment modes—baseline, L1, L2 and SM.
- `src/photonic_training/examples/train_reference.py`: a self-contained smoke
  test using synthetic data and a compact voltage-mapped reference model.
- `src/photonic_training/examples/cifar10/`: a task-level CIFAR-10 example
  using the public 32×32 voltage-mapped classifier.

## Objective

For a task loss `L_task` and voltage tensor `V`, the released implementation
uses

`L = L_task + lambda_L1 * ||V||_1 + lambda_SM * ||diff(V)||_2`.

The L2 term is applied once through decoupled AdamW weight decay
(`l2_weight_decay`), rather than being added a second time to the loss.  The SM
term is computed between adjacent mapped 32×32 blocks along the first dimension
of the voltage tensor.

### Post-training voltage map

The model is trained with the raw electrical-control tensor using
`chip.set_s(weight, mode="nophase")`.  After training, convert this raw tensor
to the deployed physical-voltage representation with `voltage_map()` before
reporting hardware quantities such as power, P99, Roff, or inter-block voltage
variation.  For a device calibrated by `phase = gamma * V**2`, use
`V_MAX = sqrt(2*pi/gamma)` and map each raw magnitude `w` as:

```python
voltage = abs(w)
if voltage > V_MAX:
    voltage = sqrt(voltage**2 - V_MAX**2)
```

`gamma` and therefore `V_MAX` are device-calibration parameters: set them from
the measured voltage-phase response of the physical chip being deployed.  This
post-training `voltage_map()` is not applied inside the model forward pass or
inside the L1/L2/SM training objective.

## Baseline and constrained task runs

Use the same task model, data split, optimizer family and evaluation protocol
for every run.  Select a configuration as follows:

| Run | L1 | L2 (AdamW weight decay) | SM |
| --- | ---: | ---: | ---: |
| Baseline | 0 | 0 | 0 |
| Sparsity-oriented | nonzero | 0 | 0 |
| Magnitude-oriented | 0 | nonzero | 0 |
| Smoothness-oriented | 0 | 0 | nonzero |

This table defines the generic method interface.  A task's baseline may retain
its established optimizer weight decay; specifically, the public CIFAR-10
baseline follows the source notebook and uses AdamW weight decay 0.01.

For each manuscript task (LiPICO-Net, Denoise-10, Denoise-17, DR²-Net and
CIFAR-10), plug its public data loader and its voltage-mapped model into the
shared `train_one_epoch` interface.  The model must return
`(prediction, voltages)`, where `voltages` has shape
`[n_blocks, 1, n_phase_shifters]` or is a list of tensors in that form.

The exact coefficient sweeps and selected representative points should be
reported from the manuscript/Supplementary Information rather than hard-coded
as universal defaults.

## Quick check

```bash
pip install -r requirements.txt
$env:PYTHONPATH = "$PWD/src"
python -m photonic_training.examples.train_reference --preset baseline
python -m photonic_training.examples.train_reference --preset combined
pytest -q
```

The reference example is only a software check; it is not a replacement for
the reported task architectures, data sets or hardware measurements.

## CIFAR-10 task-level example

The CIFAR-10 folder is a practical public validation of the same baseline and
regularization path.  It downloads only the official CIFAR-10 data set; no
checkpoints or manuscript result files are included.  For a short GPU smoke
test, run:

```bash
$env:PYTHONPATH = "$PWD/src"
python -m photonic_training.examples.cifar10.train --preset baseline --download --epochs 1 --subset-train 128 --subset-test 128
```

Then repeat with `--preset l1`, `l2`, `sm`, or `combined`.  Use no subset
arguments only for a full training run.  Matching the manuscript results also
requires the manuscript's full training schedule and selected coefficient
sweeps; this example intentionally makes no claim that a one-epoch run will
reproduce reported accuracy.

For the CIFAR baseline, the default full-run settings reproduce the available
source notebook settings: 40,000/10,000 fixed train/validation split (seed
1024), the same augmentation and normalization, batch size 1024, AdamW learning
rate 0.001, weight decay 0.01, and 10,000 epochs.  The original notebook uses
an external early-stopping helper that is intentionally not included here; the
public script reports validation and test accuracy every epoch instead.

