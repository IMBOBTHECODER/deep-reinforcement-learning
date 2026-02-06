# Evaluation Guide

## Running Evaluation

Once training is complete, evaluate the trained model with 3D visualization:

```bash
python eval.py
```

## Camera Controls During Evaluation

### Navigation
- **W** / **S**: Move camera forward/backward (along view direction)
- **A** / **D**: Strafe left/right
- **LMB (Left Mouse Button)**: Rotate camera (click and drag)

### Display Information
During evaluation, the 3D window shows:
- **Agent Position**: Current position of the quadruped (top-left)
- **Goal Position**: Target location the agent is moving toward (yellow text)
- **Distance**: Current distance to the goal
- **Step Count**: Number of simulation steps in current episode
- **FPS**: Frames per second (top-left corner)

## What to Watch For

### Good Performance
- Agent maintains stable balance on all four legs
- Smooth, coordinated walking toward goals
- Reaches goals within reasonable steps
- Minimal jerking or unnatural movements

### Common Issues
- **Wobbling/Instability**: Agent hasn't converged yet; train longer
- **Circular Motions**: Reward signal may need tuning
- **Static/No Movement**: Check that checkpoint loaded successfully
- **Crashes**: Verify simulator physics initialized correctly

## Batch Training & Evaluation Workflow

### Option 1: Train Only
```bash
python train.py
```
Trains the agent and saves checkpoints. No rendering.

### Option 2: Evaluate Only
```bash
python eval.py
```
Loads the latest checkpoint and visualizes performance with rendering.

### Option 3: Train + Evaluate
```bash
python run.py
```
Trains, then automatically evaluates when complete (original behavior).

## Tips

1. **Multiple Evaluation Runs**: You can run `python eval.py` multiple times with the same checkpoint to observe different episodes (goals spawn randomly).

2. **Comparing Models**: Save different checkpoints and evaluate each:
   ```bash
   cp checkpoint/model.pt checkpoint/model_v1.pt
   python train.py  # Train more
   python eval.py   # Evaluate latest
   ```

3. **Tweaking Rewards**: If you modify `config.py` reward parameters and retrain, use `python eval.py` to quickly test changes without waiting for full training.

4. **Recording Evaluation**: On Windows, use tools like OBS Studio to record the evaluation window.

## Configuration for Evaluation

Key settings in `config.py`:

```python
EVAL_EPISODES = 2              # Number of evaluation episodes to run
EVAL_STEPS_PER_SEC = 60        # Simulation speed (lower = faster animation)
RUN_EVALUATION = True          # Set to False to skip evaluation after training
```

## Checkpoint Management

Checkpoints are saved to: `checkpoint/model.pt`

Each training run overwrites the previous checkpoint. To keep multiple versions:

```bash
# Before training a new variant
cp checkpoint/model.pt checkpoint/model_baseline.pt

# Train your variant
python train.py

# Now you have both checkpoint/model.pt (new) and checkpoint/model_baseline.pt (old)
```

See [CONFIGURATION.md](CONFIGURATION.md) for more details on checkpoint settings.
