# Snake PPO

## Training
Start training with:

```bash
python train_ppo.py [--random-start] [--steps N]
```

* --random-start -> If provided each episode begins at a random position and heading (otherwise the snake always starts at the center of the grid)

* --steps -> N Total number of PPO training timesteps (default:1 000 000)


Example for random start and steps 2 000 000:
```bash
python train_ppo.py --random-start --steps 2000000
```

The final trained policy will be saved to:
```python
ppo_snake_cnn_final.zip
```
## Evaluation
Start evaluation with:

```bash
python evaluate_and_plot.py [--random-start] [--n_episodes M] [--model_path PATH] [--save_plot FILE]
```

* --random-start -> Run evaluation episodes from random start positions (omit to use center start).
* --n_episodes M -> Number of episodes to average over (default: 100)
* --model_path PATH -> Path to the trained model ZIP (default: ppo_snake_cnn_final.zip).
* --save_plot FILE -> Filename for the reward-curve image (default: ppo_snake_evaluation.png)

Example for evaluation with 100 episodes/games and specific source model paths and destination evaluation plot PNG:
```python
python evaluate_and_plot.py --random-start --n_episodes 100 --model_path ppo_snake_cnn_final.zip --save_plot ppo_snake_evaluation.png
```