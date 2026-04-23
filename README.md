SAR Project Commands and Options

This file lists all runnable commands and CLI options for:
- train_rl.py
- eval_runner.py

1) Training Script: train_rl.py

Basic use

Train:
  - python train_rl.py
  - python train_rl.py --mode train
Environment check only:
  - python train_rl.py --mode check

Options

--mode
  - choices: train, check
  - default: train
--device
  - choices: auto, cpu, cuda
  - default: auto
--resume
  - flag
  - when set: resume training from saved checkpoint

Examples

Fresh training on CUDA:
  - python train_rl.py --mode train --device cuda
Resume training on CUDA:
  - python train_rl.py --mode train --device cuda --resume
Validate env wrapper:
  - python train_rl.py --mode check

2) Evaluation Script: eval_runner.py

Basic usage

Compare RL and classical (default mode is both):
  - python eval_runner.py
RL only:
  - python eval_runner.py --mode rl
Classical only:
  - python eval_runner.py --mode classical

Options

--mode
  - choices: rl, classical, both
  - default: both

--train-file
  - path to training script to import
  - default: train_rl.py

--episodes
  - int
  - default: 100

--device
  - choices: auto, cpu, cuda
  - default: auto

--no-prompt
  - flag
  - run without Enter prompts or renderer

--render-every
  - int
  - default: 30
  - interactive rendering cadence (higher is less lag)

--show-final-only
  - flag
  - skip live rendering and only show final snapshot per episode

--final-hold-seconds
  - float
  - default: 4.0
  - only used when --no-prompt and --show-final-only are active

--compare-classical
  - deprecated compatibility flag
  - if provided with --mode rl, it forces both

--seed-base 
  - int
  - default: training SEED from train file
  - sets first episode seed; episode i uses seed_base + i

--eval-persons
  - int
  - default: from training file EVAL_ACTIVE_PERSONS

--eval-obstacles
  - int
  - default: from training file EVAL_ACTIVE_OBSTACLES

--model-path
  - repeatable
  - accepts with or without .zip
  - for mode rl or both, each --model-path adds one RL model to evaluate

3) Useful Command Recipes

A) Classical only, one episode, seed 2025, 3 people/3 obstacles, final snapshot

python eval_runner.py --mode classical --episodes 1 --start-seed 2025 --eval-persons 3 --eval-obstacles 3 --show-final-only

B) Classical only, fast/headless (no prompts)

python eval_runner.py --mode classical --episodes 10 --start-seed 2025 --eval-persons 3 --eval-obstacles 3 --no-prompt

C) RL only using best checkpoint

python eval_runner.py --mode rl --model-path checkpoints\\best_model.zip --episodes 100 --start-seed 2025 --no-prompt

D) RL vs classical for one model

python eval_runner.py --mode both --model-path checkpoints\\best_model.zip --episodes 100 --start-seed 2025 --no-prompt

E) Compare multiple RL models and classical

python eval_runner.py --mode both --model-path checkpoints\\best_model.zip --model-path ppo_swarm_agent_v9.zip --model-path checkpoints_exp_1m\\best_model.zip --episodes 100 --start-seed 2025 --no-prompt

F) Reduce lag in interactive rendering

python eval_runner.py --mode classical --episodes 1 --start-seed 2025 --eval-persons 3 --eval-obstacles 3 --render-every 60

4) Output Notes

- train_rl.py writes live logs to train_log.txt.
- eval_runner.py saves comparison plots to eval_plots/ when plotting paths are generated.
