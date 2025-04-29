# Snake Deep Q-Learning

## Training
Start training with:

```bash
python agent.py [--mode train] [--games N] [--model M]
```

* --games N -> Total number of games to train on (default: 10000)
* --model M -> Model to load from after M games (default: None)



Example for laod model after 1000 games and learn additionally 1000 games:
```bash
python train_ppo.py --mode train --iterations 1000 --model 1000
```

The final trained model will be saved to:
```python
model/model.pth
```
## Evaluation
Start evaluation with:

```bash
python agent.py [--mode eval] [--games N] [--model M]
```

* --games N -> Total number of games to evaluate on (default: 100)
* --model M -> Model to load from after M games (default: evaluate all models in model/)

Example for evaluation for model after 1000 training games with 100 games:
```python
python agent.py --mode eval --games 100 --model 1000
```