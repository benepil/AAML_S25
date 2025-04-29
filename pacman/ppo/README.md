# Snake PPO

## Training
Start training with:

```bash
python train_ppo.py [--steps N] [--n-envs E] [--seed S]
```
* --steps N -> Total number of PPO training timesteps (default: 3000000)
* --n-envs E -> Number of parallel environments (default: 8)
* --seed S -> Base random seed for reproducibility (default: 0)

Example (1 000 000 steps, 8 envs, seed 42):
```bash
python train_ppo.py --steps 1000000 --n-envs 8 --seed 42
```

The final trained policy will be saved to:
```python
ppo_pacman_final.zip
```
## Evaluation
Start evaluation with:

```bash
python evaluate_and_plot.py [--model_path PATH] [--n_episodes M] [--save_plot FILE]
[--save_plot FILE]
```
* --model_path PATH -> Path to the saved model ZIP (default: ppo_pacman_final.zip)
* --n_episodes M -> Number of episodes to average over (default: 100)
* --save_plot FILE -> Filename for the reward-curve PNG (default: ppo_pacman_eval.png)

Example (evaluate 200 episodes, custom model and output):
```python
python evaluate_and_plot.py --model_path ppo_pacman_final.zip --n_episodes 200 --save_plot ppo_pacman_eval_200.png
```