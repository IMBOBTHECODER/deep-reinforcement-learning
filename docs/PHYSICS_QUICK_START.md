# Quick Start: Physics Configuration Presets

**Just want to train? Pick a preset below, copy the config, and run!**

---

## 🚀 Preset 1: Speed (Fastest)
For quick testing and debugging.

```python
# config.py
ACTUATOR_RESPONSE_TIME = 0.0
FRICTION_MODEL = "simple"
TRACK_ENERGY_CONSUMPTION = False
```

**Training time**: 30 episodes to balance  
**Realism**: Low  
**When to use**: Debugging, rapid iteration  

---

## ⚖️ Preset 2: Balanced (Recommended ✅)
Best default for most training.

```python
# config.py
ACTUATOR_RESPONSE_TIME = 0.01
FRICTION_MODEL = "coulomb+viscous"
FRICTION_COEFFICIENT_STATIC = 0.9
FRICTION_COEFFICIENT_KINETIC = 0.85
FRICTION_VISCOUS_DAMPING = 0.05
TRACK_ENERGY_CONSUMPTION = True
MOTOR_EFFICIENCY = 0.80
```

**Training time**: 32 episodes to balance  
**Realism**: High  
**When to use**: Most training runs, research  

---

## 🎯 Preset 3: Realism (Highest Fidelity)
For validation and real-world transfer.

```python
# config.py
ACTUATOR_RESPONSE_TIME = 0.015
FRICTION_MODEL = "coulomb+viscous"
FRICTION_COEFFICIENT_STATIC = 0.92
FRICTION_COEFFICIENT_KINETIC = 0.85
FRICTION_VISCOUS_DAMPING = 0.06
TRACK_ENERGY_CONSUMPTION = True
MOTOR_EFFICIENCY = 0.78
```

**Training time**: 35 episodes to balance  
**Realism**: Very High  
**When to use**: Real robot transfer, validation papers  

---

## 📚 Learn More About Each Feature

| Feature | Doc | What It Does |
|---------|-----|-------------|
| **Actuator Lag** | [ACTUATOR_LAG.md](ACTUATOR_LAG.md) | Motor response time (10ms typical) |
| **Friction** | [FRICTION_MODELS.md](FRICTION_MODELS.md) | Stick-slip, realistic foot grip |
| **Energy** | [ENERGY_TRACKING.md](ENERGY_TRACKING.md) | Power consumption monitoring |

---

## ⚙️ Common Tweaks

### For Icy Surface
```python
FRICTION_COEFFICIENT_STATIC = 0.3
FRICTION_COEFFICIENT_KINETIC = 0.2
```

### For Muddy Terrain
```python
FRICTION_COEFFICIENT_STATIC = 1.0
FRICTION_COEFFICIENT_KINETIC = 0.95
FRICTION_VISCOUS_DAMPING = 0.1
```

### For Robot Hardware (Dynamixel XM430)
```python
ACTUATOR_RESPONSE_TIME = 0.05  # 50ms response time
MOTOR_EFFICIENCY = 0.75        # From datasheet
```

---

## 🏃 Train Now!

1. **Choose a preset** above (Preset 2 recommended)
2. **Edit config.py** with the parameters
3. **Run training**:
   ```bash
   python train.py
   ```

That's it! The physics engine handles the rest automatically.

---

## ✅ Verify It's Working

After training starts, check that:
- Agent is moving (not stuck)
- Gait looks smooth (not jerky)
- Rewards increase over time
- No error messages

If something looks wrong, see **Diagnostics** in the feature docs above.

