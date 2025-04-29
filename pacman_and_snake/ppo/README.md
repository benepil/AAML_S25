# Multi-Game PPO (Snake + Pac-Man)

## Training

Launch multi-game PPO training with:

```bash
python train_ppo.py [--steps N] [--pacman-pretrain P] [--n-envs E] [--seed S]
```
* --steps N -> Total timesteps for the entire run (pretrain + mixed). Default: 3 000 000
* --pacman-pretrain P -> Number of timesteps to train on Pac-Man only before mixing with Snake. Default: 250 000
* --n-envs E -> Number of parallel environments. Default: 8
* --seed S -> Base random seed for reproducibility. Default: 0


Example (pretrain 250 K → mixed 250 K, 8 envs, seed 0):
```bash
python train_ppo.py --steps 500000 --pacman-pretrain 250000 --n-envs 8 --seed 0
```

Another example:
```bash
python train_ppo.py --steps 500000 --pacman-pretrain 250000 --n-envs 8 --seed 42
```
This will: 
* Pre-train a shared CNN + multi-head PPO policy on Pac-Man for 250 K steps
* Switch to a mixed Snake + Pac-Man training for the remaining 250 K steps, alternating tasks across 8 workers.

The final trained policy will be saved to:
```python
ppo_multi_final.zip
```

## Evaluation
Run evaluation separately on Snake and Pac-Man with:
```bash
python evaluate_and_plot.py [--model_path PATH] [--n_episodes M] [--save_plot FILE]
```

* --model_path PATH -> Path to the trained multi-game PPO ZIP. Default: ppo_multi_final.zip
* --n_episodes M -> Number of episodes per task. Default: 100
* --save_plot FILE -> Filename for the combined reward curve PNG. Default: ppo_multi_eval.png

Example (100 episodes, default model & plot file):
```bash
python evaluate_and_plot.py --n_episodes 100 --save_plot ppo_multi_eval.png
```