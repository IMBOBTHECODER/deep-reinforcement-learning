# Physics Engine & Rigid Body Dynamics

## Overview

This document describes the **advanced physics simulation** with **quaternion-based orientation**, **rigid body dynamics**, **agent-centered world coordinate system**, and professional-grade contact modeling for stable quadruped balance learning.

**Location:** [source/physics.py](../../source/physics.py)

---

## Part 1: Agent-Centered World Coordinates

### Simple Explanation

Imagine you're standing in a room. You could describe everything relative to the room (global coordinates), or you could describe everything relative to yourself (your perspective).

**Global view:** "The door is at (100, 50), I'm at (70, 40), so the door is 30 units forward and 10 units right."

**Your view:** "The door is at (30, 10) relative to me" - which is simpler and always works no matter where you move.

Our agent uses the second approach: **the agent is always at the center, and everything is described relative to the agent.**

### Why This Matters for Learning

**Without agent-centered coords:**
- Agent learns "go to position (87, 45)" - specific to that episode
- If goal moves to (100, 60), agent must re-learn
- Different episodes = different problems

**With agent-centered coords:**
- Agent learns "go forward and right" - general principle
- Works for any goal location
- **Same policy works everywhere** - we only need to train once!

**Real example:** A dog learns "walk toward things" not "walk to position X". The dog's skill transfers to new environments.

### How It Works

```python
# Imagine:
# - Agent at global position (70, 40, 0.5)
# - Goal at global position (100, 50, 2)
# - In agent-centered frame, agent is always at (0, 0, 0)

# Goal becomes relative:
goal_relative = [100-70, 50-40, 2-0.5] = [30, 10, 1.5]

# Agent only sees: "Goal is 30 units forward, 10 right, 1.5 up"
# Agent doesn't know where it is in the world - only where goal is!
```

### Observation Structure

The agent receives:
```
Observation = [
    joint_angles,           # What are my leg positions? (12D)
    joint_velocities,       # How fast are my legs moving? (12D)
    foot_contact,           # Which feet touch ground? (4D)
    orientation,            # Which way am I facing? (3D)
    [0, 0, 0],             # My COM is at origin (3D)
    goal_relative,          # Where is the goal relative to me? (3D)
]
# Total: 38D observation, all centered on agent
```

---

## Part 2: Advanced Physics Engine

### 1. Quaternion-Based Orientation

**The Problem with Standard Rotations:**
Try rotating your arm in 3D. If you tilt it too far (90° pitch), you lose control - a degree of freedom disappears. Engineers call this "gimbal lock". It's terrible for AI because the agent gets stuck.

**The Solution - Quaternions:**
A quaternion `(w, x, y, z)` is a 4-number trick that represents rotations perfectly - no singularities, no stuck points, ever.

**Why it matters:**
- Agent can rotate freely without losing control
- Smooth, continuous rotations (neural networks love this)
- No mathematical singularities (no "stuck" angles)
- Slightly faster than 3×3 rotation matrices

```python
# Simple usage:
q = Quaternion.from_euler(pitch=45°, yaw=30°, roll=0°)
# Works smoothly at ANY angle - no gimbal lock!

# Rotate a vector (like rotating the up direction)
gravity_in_body_frame = q.rotate_vector([0, 0, -9.81])

# Combine rotations smoothly
q_total = q1 * q2
```

**Real example:**
Dog tilts head backwards while jumping. With Euler angles, tilt would cause gimbal lock. With quaternions, it works perfectly.

### 2. Full Rigid Body Dynamics

**Simple Idea:** Everything moves two ways - linearly (in a direction) and rotationally (spinning). Both follow Newton's laws.

**Linear Motion (Straightforward):**
```
Force applied → Object accelerates
Acceleration happens → Velocity changes
Velocity exists → Position changes

F = m * a           (Force = Mass × Acceleration)
a = F/m             (Rearranged: Heavy things accelerate slower)
v = v + a*dt        (Velocity updates from acceleration)
x = x + v*dt        (Position updates from velocity)

Real example - 5kg dog, 50N gravity:
- Acceleration = 50 / 5 = 10 m/s²
- After 0.1 sec: v = 0 + 10*0.1 = 1 m/s (falling faster)
- After 0.2 sec: v = 1 + 10*0.1 = 2 m/s (falling even faster)
```

**Angular Motion (Rotation - More Complex):**
```
τ = I * α + ω × (I*ω)

Plain English version:
- τ (tau) = torque you apply (like pushing a spinning top)
- I = resistance to rotation (heavy objects hard to spin)
- α (alpha) = angular acceleration (how fast rotation speed changes)
- ω × (I*ω) = GYROSCOPIC EFFECT (spinning resists tipping!)

The gyroscopic term is the magic:
- Spinning top resists gravity trying to tip it
- Our agent can use body spin to balance while recovering from a fall
- The faster the spin, the more it resists tipping

Real example - spinning top:
1. Top spins fast (large ω)
2. Gravity tries to tip it over (τ acts)
3. But gyroscopic term ω × (I*ω) resists
4. Result: Top stays upright!

For the agent:
- Heavy limbs are hard to move (large I)
- Quick balance corrections require large τ
- Body can "absorb" fall energy by spinning (gyroscope effect)
```

**Integration (How it All Works Together):**
```
Simplified step-by-step:
1. Motors apply torques to joints
2. Forces (gravity, contacts) act on body
3. Accelerations computed from forces/torques
4. Velocities updated from accelerations
5. Positions/orientations updated from velocities
6. Repeat every 0.01 seconds
```

### 3. Inertia Tensor (Moment of Inertia)

**Simple Concept:** Inertia is "resistance to rotation" - how hard it is to spin something.

**Analogy:** 
- Light stick - easy to spin fast
- Heavy stick - hard to spin fast
- Longer stick - even harder to spin

Mathematically, inertia depends on mass AND how far that mass is from the rotation axis.

**For Our Quadruped:**
The body (torso) has different inertia for different rotation axes:

```
Torso dimensions: 0.5m long, 0.2m tall, 0.3m wide, ~1.0 kg

I_roll (roll around length): ~0.0083 kg⋅m²    (easiest to roll)
I_pitch (pitch around width): ~0.0208 kg⋅m²   (medium)
I_yaw (yaw around height): ~0.0208 kg⋅m²      (medium)

Comparison - rotating the torso:
- Roll (lean side-to-side): Easy, fast recovery
- Pitch (tilt forward-back): Medium, hardest to recover
- Yaw (spin): Medium, fastest standing spin
```

**Why it matters:**
```
Balance recovery requires overcoming inertia:
- Agent falls forward → Needs torque to pitch back
- Larger I_pitch → Harder to recover → Needs stronger motors
- Heavier body → Larger inertia → Harder to adjust posture
```

**Practical tuning:**
- Too small I: Agent can flip easily (instability)
- Too large I: Agent can't correct posture (slow recovery)
- Must match motor strength to inertia

### 4. Contact-Dependent Gravity

**The Idea:** When standing on ground, you don't "feel" gravity as much. But when jumping, gravity pulls you down fast. Physics should match this.

**Implementation:**
```
gravity_factor = 1.0 - (number_of_feet_in_contact / 4.0)

Examples:
- All 4 feet on ground: gravity_factor = 0.0  (balanced, no gravity effect)
- 3 feet: gravity_factor = 0.25
- 2 feet: gravity_factor = 0.50 (half support)
- 1 foot: gravity_factor = 0.75 (barely supported)
- 0 feet: gravity_factor = 1.0  (free falling)

Applied as:
actual_gravity = [0, 0, gravity_factor * -9.81]  (upward pull is reduced when supported)
```

**Why this works:**
- Agent doesn't fight gravity when stable → Reward for standing still
- Agent experiences full gravity when falling → Realistic falling
- Partial support means partial effect → Natural intermediate state

**Real-world physics:**
When you stand still, ground pushes up to cancel gravity. Our model skips the ground reaction force detail and just scales gravity. It's simpler and works great for learning.

### 5. Spring-Damper Contact Model

**What is Ground Contact?**
Feet don't actually sink into the ground. Instead, ground pushes back with a force that depends on:
1. How deep the foot penetrates (spring)
2. How fast the foot is moving downward (damper/energy loss)

**The Formula:**
```
F_contact = K * penetration - C * velocity

Where:
- K = 0.5 (stiffness) - how hard the ground pushes back
- C = 0.2 (damping) - energy dissipation (reduces bouncing)
- penetration = max(0, ground_level - foot_z)  (how deep foot is in ground)
- velocity = foot velocity downward (m/s)
```

**Intuition - Like a Bouncy Ball:**
```
DROP a ball:
1. Ball hits ground (penetration increases)
2. Ground pushes back (spring term: K * penetration)
3. Ball tries to bounce (velocity reverses)
4. Damping absorbs bounce energy (C * velocity)
5. Result: Ball bounces less than you'd expect (realistic!)

Same for foot:
- Foot hits ground → Spring pushes up
- Foot slows down → Damper absorbs energy
- No more bouncing → Foot stays stable
```

**Parameters (How to Tune):**
```
K = stiffness (0.5):
  - Higher → ground feels harder, less sinking
  - Lower → ground feels softer, more sinking
  - Too high → foot oscillates wildly
  - Too low → foot sinks too deep

C = damping (0.2):
  - Higher → more energy lost, less bouncing
  - Lower → energy preserved, more bouncing
  - Our value (0.2) balances stability and realism
```

**What Actually Happens During Simulation:**
```python
def compute_contact_force(foot_z, velocity_z, ground_level=0.0):
    penetration = max(0, ground_level - foot_z)
    
    if penetration > 0:
        # Spring force: pushes foot UP out of ground
        spring_force = 0.5 * penetration
        
        # Damper force: opposes downward motion
        damper_force = 0.2 * velocity_z
        
        # Net contact force
        contact_force = spring_force - damper_force
        return contact_force
    else:
        return 0  # No contact
```

### 6. Sequential Impulses Constraint Solver

**What's the Problem We're Solving?**
When two objects collide (foot hits ground), what happens?
- They can't penetrate each other
- They might stick together (friction)
- Forces are applied instantly (impulses)

Traditional solvers use small timesteps and hope for stability. We use a better method: **iterative impulses**.

**Simple Idea - Bounce Ball Example:**
```
1. Ball hits wall
2. Wall can't move ball through it
3. Wall applies instant "impulse" to reverse velocity
4. Ball bounces off

Real formula:
J_normal = v_rel · normal  (velocity toward wall)
λ = -J_normal / mass       (impulse needed to stop)
v_new = v + λ               (velocity after bounce)
```

**For Friction (Tangent Direction):**
```
Coulomb Cone Rule (simple friction):
max_friction = μ * normal_impulse

Examples:
- Dry ground: μ = 0.8  (high friction, hard to slide)
- Ice: μ = 0.1  (low friction, easy to slide)

What happens:
1. Normal impulse prevents penetration
2. Friction impulse opposes sliding (up to max_friction)
3. If push hard enough sideways, it overcomes friction and slides
4. If push gently, foot stays stuck
```

**How the Solver Works (Iterative):**
```
FOR each contact (4 contacts = 4 feet):
  1. Calculate relative velocity at contact
  2. Apply normal impulse (prevent penetration)
  3. Apply friction impulse (prevent sliding, up to cone)
  4. Update velocities

Repeat 10 times:
  - First iteration fixes biggest problems
  - Later iterations refine small issues
  - More iterations = more stable = slower

We use ~10 iterations for balance between speed and stability
```

**Benefits:**
- Stable (won't explode if you miss collision)
- Realistic friction (sliding vs sticking)
- Works with joints (internal constraints)
- Time-efficient

### 7. Joint Constraints (Motor Control)

**What are Joint Constraints?**
Joints connect body parts. Motors apply torques to rotate joints. But unbounded speeds would cause instability.

**Maximum Joint Speed:**
Each joint has a speed limit: **±10 rad/s** (about 600 RPM)

**Why the Limit?**
```
Unrealistic speeds:
- >20 rad/s: Motors can't achieve this
- >15 rad/s: Control becomes unstable
- 10 rad/s: Real biomorphic limit

Without limit:
- Agent learns unrealistic movements
- Numerical instability (huge velocities break solver)
- Looks unnatural (twitchy, jerky)

With limit:
- Movements look natural
- Solver stays stable
- More realistic training
```

**How it Works (Simplified):**
```python
# Each timestep:
new_velocity = current_velocity + (applied_torque - damping) * dt

# But don't allow unrealistic speeds
if new_velocity > 10:  # rad/s
    new_velocity = 10

if new_velocity < -10:
    new_velocity = -10

# Update joint angle
new_angle = current_angle + new_velocity * dt
```

**Real Example - Leg Swing:**
```
Hip motor applies torque to swing leg forward
- Desired speed: 5 rad/s (50% of max)
- Joint accelerates
- Reaches 5 rad/s, maintains it (hits speed limit)
- Motor switches to slow down
- Joint decelerates smoothly
- Result: Natural leg swing without overshooting
```

### 8. Foot-Ground Contact Detection

**The Question:** Is the foot touching the ground? Answer: "kind of"

**Why not Yes/No?**
Transitions are the problem:
```
Binary approach (foot touching or not):
- Foot at 0.00m: TOUCHING
- Foot at 0.01m: NOT TOUCHING (sudden jump!)
- Neural network hates sudden jumps

Continuous approach (0.0 to 1.0):
- Foot at 0.00m: 1.0 (full contact)
- Foot at 0.02m: 0.6 (light contact)
- Foot at 0.04m: 0.2 (barely touching)
- Foot at 0.05m: 0.0 (liftoff complete)
- Neural network loves smooth transitions!
```

**The Algorithm:**
```python
def detect_contact(foot_z, ground_level=0.0, threshold=0.05):
    # How deep is the foot in/above ground?
    distance = foot_z - ground_level
    
    if distance <= 0:
        return 1.0  # Deep contact
    elif distance <= 0.05:  # In threshold zone
        # Fade from 1.0 to 0.0 over 5cm
        return (0.05 - distance) / 0.05
    else:
        return 0.0  # No contact (too high)
```

**Threshold = 0.05m (5cm):**
- Period when foot is touching but liftoff started
- Short enough that transitions are quick
- Long enough for smooth gradients in learning

**Real Example - Walking Gait:**
```
Frame 1: Foot lands (z = 0.00): contact = 1.0
Frame 2: Foot still down (z = 0.01): contact = 0.8
Frame 3: Foot lifting (z = 0.04): contact = 0.2
Frame 4: Foot in air (z = 0.06): contact = 0.0

Neural network sees smooth value, learns smooth walking!
```

### 9. Contact Friction Cone

**Real-World Friction:**
If you push something with a horizontal force:
- Small push → It doesn't slide (friction holds)
- Bigger push → It starts to slide (overcame static friction)
- Max static friction = μ_s × normal_force

**In Our Physics:**
When a foot touches ground:
- Ground applies normal force (pushing up)
- Foot can't slide sideways more than: μ × normal_force

**Coulomb Friction Model (Simple and Effective):**
```
friction_limit = μ * normal_impulse

Where:
μ = 1.0 (rubber on concrete - high grip)
Or μ = 0.1 (ice on steel - low grip)

What happens:
1. Foot lands (normal force = N)
2. Sideways push tries to slide foot (F_side)
3. Friction limit = μ*N
4. If F_side < μ*N: Foot sticks (no sliding)
5. If F_side > μ*N: Foot slides (overcame friction)

Real example - walking:
- Foot lands with weight (N = ~50N)
- Friction limit = 1.0 * 50N = 50N
- Agent pushes sideways with 30N
- 30N < 50N → Foot sticks → Walking works!
```

**Why This Matters:**
- Prevents unrealistic "skating" gaits
- Makes slopes harder (gravity pulls sideways)
- Encourages realistic foot placement

---

## Part 3: Reward Function (Balance-Focused)

**Simple Idea:** Tell the agent what you want it to learn, using a reward signal:
- Do good thing → Positive reward (happy!)
- Do bad thing → Negative reward (unhappy)
- Agent learns to maximize happiness

**The Recipe:**
```
Total Reward = Goal Progress + Stability Bonus - Balance Penalties - Energy Cost

Real example (one step):
Goal Progress: 0.8  (got closer to goal)
Stability:     0.2  (3 feet on ground)
Tilt Penalty: -0.5  (tilting too much)
Energy Cost:  -0.3  (motors working hard)
───────────
Total Reward:  0.2  (good step, keep going!)
```

### 1. Goal Reward (Making Progress)

**What:** How close are you to the target?

```
distance_to_goal = ||current_position - target_position||

if distance < 0.1m:  (very close!)
    reward = 0.8    (strong positive reward)
else if distance < 0.5m:  (reasonable)
    reward = 0.4
else:
    penalty = -distance × 0.1  (further away = bigger penalty)

Real example:
- Goal is 1m away → reward = -0.1 (penalty)
- Goal is 0.3m away → reward = +0.4 (progress!)
- Goal is 0.05m away → reward = +0.8 (almost there!)
```

**Psychology:**
The agent learns: "Getting closer to goal = good, being far = bad"

### 2. Contact Reward (Stay Stable)

**What:** Are your feet touching the ground?

```
contact_reward = num_feet_in_contact × 0.1

Examples:
- All 4 feet: 0.4 reward  (stable!)
- 3 feet: 0.3 reward      (mostly stable)
- 2 feet: 0.2 reward      (balance mode)
- 0 feet: 0.0 reward      (in air)
```

**Psychology:**
The agent learns: "Stay on the ground for stability bonus"

### 3. Tilt Penalty (Don't Fall Over)

**What:** How tilted are you?

```
Max allowed tilt: 30° (0.52 radians)

if tilt > 30°:
    penalty = -5.0  (strong punishment!)
else:
    penalty = 0     (acceptable)

Real example:
- Tilt 10°: No penalty (OK)
- Tilt 40°: -5.0 penalty (stop falling!)
- Tilt 60°: -5.0 penalty (already being punished)
```

**Psychology:**
The agent learns: "Stay upright or suffer big penalties"

### 4. Energy Cost (Be Efficient)

**What:** How hard are motors working?

```
energy_cost = sum(|motor_torques|) × 0.01

Examples:
- Gentle movement (torques = 5): -0.05 cost
- Hard movement (torques = 50): -0.5 cost
```

**Psychology:**
The agent learns: "Smooth, efficient movement gets better rewards than jerky, wasteful movement"

### Putting It Together

```
One timestep example:
────────────────────────────────────────
Position: 2m from goal (down from 2.5m)
Contact: 3 feet
Tilt: 15°
Torques: 30 units applied
────────────────────────────────────────

Rewards:
  Goal progress: -0.2 (closer! worth -0.2 penalty... wait)
  Actually: 0.4 (made progress toward goal)
  Contact: 0.3 (3 feet, 3×0.1)
  Tilt: 0 (15° < 30°, OK)
  Energy: -0.3 (30 units × 0.01)
  ─────────
  Total: +0.4 reward

Agent thinks: "Good step! Keep walking like this!"
```

---

## Part 4: Stability Analysis

**The Question:** When does a quadruped fall over?

### Standing Stable (The Ideal)

**Conditions for Stable Standing:**
```
1. All 4 feet on ground
   → Gravity doesn't pull down (gravity_factor = 0)
   → Weight fully supported

2. Center of mass above foot area
   → Imagine 4 corners where feet are
   → Draw a rectangle connecting them
   → COM is inside this rectangle
   → Balanced!

3. Joint velocities are low
   → Joints aren't vibrating
   → Body isn't twitching
   → Standing still

Result: STABLE - Agent can stand indefinitely
```

**Real Example - Standing Dog:**
```
Feet at positions:
  Front-left: (0.1, 0)
  Front-right: (-0.1, 0)
  Back-left: (0.1, -0.3)
  Back-right: (-0.1, -0.3)

Foot rectangle: From (-0.1, -0.3) to (0.1, 0)

COM at (0, -0.15): INSIDE rectangle ✓ STABLE

COM at (0.2, 0): OUTSIDE rectangle ✗ FALLING!
```

### Tilted (Risky Zones)

**Danger Zones:**
```
Slight tilt (15°-30°):
  - Feet still on ground
  - COM still over base
  - Unstable but recoverable
  - High risk of falling

Medium tilt (30°-45°):
  - Body at dangerous angle
  - COM moving toward edge
  - Very hard to recover
  - Usually falls

Critical tilt (>45°):
  - Body nearly on side
  - No recovery possible
  - FALLING confirmed
```

**Recovery Requirements:**
To recover from a tilt:
1. Feet must remain in contact (push against ground)
2. Motors must apply correcting torque
3. Torque must overcome gyroscopic effects
4. All must happen in fractions of a second

### Critical Angles (Thresholds)

```
Stable: ±0.3 radians (±17°)
  - Can stand easily
  - Recovery very likely

Marginal: ±0.3 to ±0.5 radians (±17° to ±29°)
  - Wobbling
  - Recovery possible but risky

Falling: >±0.5 radians (>±29°)
  - Definitely falling
  - No recovery possible
  - Episode ends

Our penalty kicks in at ±0.5 rad to prevent extreme tilts
```

**Real Example - Recovery During Tilt:**
```
Frame 1: Tilt = 0°, Standing
Frame 2: Toe catches ground → Tilt = 10°
Frame 3: Motors apply correcting torque
Frame 4: Tilt = 5° (improving!)
Frame 5: Tilt = 0° (recovered!)

Total recovery time: ~0.04 seconds (very fast!)

Agent must learn this instinctively through trial-and-error
```

---

## Part 5: Class Reference

**Note:** This is a reference section. For detailed theory, see earlier parts. This explains "what problem does each class solve?"

### `Quaternion` - Rotation Without Gimbal Lock

**Problem it solves:** How do you represent rotations without "getting stuck" at certain angles?

**What it does:**
- Stores orientation as `(w, x, y, z)` instead of angles
- Prevents gimbal lock (no stuck angles)
- Allows smooth interpolation between rotations

**Common operations:**
```python
# Create from Euler angles
q = Quaternion.from_euler(pitch=0.1, yaw=0.2, roll=0)

# Rotate a vector (like rotating "up" with body)
rotated = q.rotate_vector(np.array([0, 0, 1]))

# Convert back to Euler
pitch, yaw, roll = q.to_euler()

# Combine rotations smoothly
q_combined = q1 * q2
```

### `RigidBody` - The Quadruped's Body Physics

**Problem it solves:** How does gravity, forces, and torques affect body motion?

**What it does:**
- Tracks position, velocity, orientation
- Applies gravity and contact forces
- Integrates physics equations each timestep
- Computes how body accelerates and rotates

**Typical flow:**
```python
body = RigidBody(
    mass=5.0,                        # 5kg dog
    pos=[0, 0, 0.3],                # Start 30cm above ground
    orientation=Quaternion(...)      # Upright orientation
)

# Each timestep:
body.add_force([0, 0, -50])      # Gravity (5kg × 10 m/s²)
body.add_contact_force([10, 0, 40], point=[0, 0, -0.15])  # Foot pushes
body.integrate(dt=0.02)          # Update for 20ms

# Check results
print(body.pos)              # New position
print(body.linear_vel)       # How fast falling
print(body.angular_vel)      # How fast spinning
```

### `ContactManifold` - Foot-Ground Interaction

**Problem it solves:** How do feet bounce off/stick to ground?

**What it does:**
- Detects when foot touches ground
- Computes spring-damper forces
- Applies friction using Coulomb cone
- Prevents feet from sinking through ground

**Real example - foot landing:**
```python
contact = ContactManifold(
    body_a=quadruped_body,      # Our foot/body
    body_b=None,                # Ground is "static"
    pos=[0, 0, 0.02],          # Contact point position
    normal=[0, 0, 1],          # Ground pushes upward
    penetration=0.02,          # Foot sank 2cm
    friction_coeff=0.8         # Rubber on ground
)

# Solve forces
forces = contact.solve_impulse(dt=0.02)
# Result: Spring force pushes foot up, damping slows impact
```

### `PhysicsEngine` - Main Orchestrator

**Problem it solves:** How do all pieces work together for a quadruped?

**What it does:**
- Manages rigid body
- Manages all 4 foot contacts
- Applies motor torques to joints
- Integrates everything 50 times per second (20ms per frame)

**Typical training loop:**
```python
physics = PhysicsEngine(gravity=9.8)

# Agent decides: apply these motor torques
motor_torques = [1.0, -0.5, ..., 0.3]  # 12 values

# Physics engine processes it:
physics.apply_motor_torques(motor_torques)
physics.step(dt=0.02)

# Get results for learning
com_pos = physics.body.pos                    # Where is body?
foot_contacts = physics.foot_contact_states   # Which feet touch?
joint_angles = physics.joint_angles           # What angles are joints at?

# These become observations that train the neural network
```

---

## Part 6: Simulation Parameters (Config)

**What these numbers mean:**  Each parameter tunes how realistic the simulation is. Different values create different behaviors and learning speeds.

### Fundamental Physics

```python
GRAVITY = 9.8  # m/s² (Earth gravity - use this for realism!)
DT = 1/50      # 0.02 seconds per simulation step

Why 50 Hz?
- 20ms per step is enough for smooth physics
- Faster is more accurate but slower
- Slower causes instability (missing collisions)
```

### Joint Motor Control

```python
MAX_TORQUE = 5.0      # Maximum force motors can apply (N⋅m)
JOINT_DAMPING = 0.01  # Resistance to motion (stabilizes)

Real example - leg swinging:
- Agent wants to swing leg: applies MAX_TORQUE
- Damping resists: proportional to velocity
- Result: Smooth swing, not sudden jerk
```

### Ground Contact Properties

```python
GROUND_FRICTION = 0.8  # How sticky is ground? (0.0=ice, 1.0=sticky)
GROUND_LEVEL = 0.0     # Where is the ground? (Z-coordinate)

Friction examples:
- 0.1 = Ice skating (hard to grip)
- 0.5 = Regular floor (normal)
- 0.8 = Rubber on concrete (very grippy)
```

### Spring-Damper Contact Model

```python
CONTACT_STIFFNESS = 0.5     # How stiff the collision response is
CONTACT_DAMPING = 0.2       # How much bounce energy is lost
FOOT_HEIGHT_THRESHOLD = 0.05 # Fade-in zone for contact (5cm)

Practical tuning:
- Stiffness too high:   Ground feels hard, feet oscillate wildly
- Stiffness too low:    Feet sink too deep
- Damping too high:     No bounce (sticky feet)
- Damping too low:      Bouncy feet (unstable)

Our values (0.5 / 0.2) balance stability and realism
```

### Motor Physics

```python
SEGMENT_LENGTH = 0.1   # Length of each leg segment (m)
MAX_JOINT_VELOCITY = 10 rad/s  # Speed limit per joint

Why limit joint velocity?
- Real motors can't go infinitely fast
- Prevents unrealistic twitching
- Helps neural network learn natural motion
```

---

## Part 7: Physics Tuning Guide - How to Get Different Behaviors

**Purpose:** These parameters let you shape how the simulation behaves. Change them to experiment!

### Scenario 1: Your Dog Is Too Floppy (Wobbles/Falls Easily)

**Symptoms:**
- Falls over from small bumps
- Wiggles excessively
- Can't stand still

**What to try:**
```python
# Increase body resistance to tilting
I_tensor = np.diag([1.0, 2.0, 1.5])  # Was [0.5, 1.2, 0.9]

# Increase ground stiffness (firmer collision response)
CONTACT_STIFFNESS = 1.0  # Was 0.5

# Increase friction (better grip)
GROUND_FRICTION = 1.0  # Was 0.8

# More joint damping (smoother motion)
JOINT_DAMPING = 0.02  # Was 0.01
```

### Scenario 2: Your Dog Is Too Stiff (Can't Learn Movement)

**Symptoms:**
- Moves jerkily
- Can't adapt to terrain
- Slow learning

**What to try:**
```python
# Reduce inertia (easier to rotate)
I_tensor = np.diag([0.3, 0.8, 0.6])  # Was [0.5, 1.2, 0.9]

# Reduce damping (snappier response)
JOINT_DAMPING = 0.005  # Was 0.01

# Reduce ground damping (bouncier)
CONTACT_DAMPING = 0.1  # Was 0.2
```

### Scenario 3: Your Dog Falls Through The Ground

**Symptoms:**
- Feet sink through floor
- Unstable contact forces

**What to try:**
```python
# Increase contact stiffness (ground pushes harder)
CONTACT_STIFFNESS = 1.0  # Was 0.5

# Reduce contact damping (allow more bounce)
CONTACT_DAMPING = 0.1  # Was 0.2

# Check threshold isn't too small
FOOT_HEIGHT_THRESHOLD = 0.1  # Was 0.05 (bigger tolerance zone)
```

### Scenario 4: Training Is Too Slow

**Symptoms:**
- Takes forever to learn
- Agent stuck

**What to try:**
```python
# Reduce physics accuracy (faster simulation)
DT = 1/30  # Was 1/50 (fewer steps per second)

# Reduce motor damping (agents can move faster)
JOINT_DAMPING = 0.005  # Was 0.01

# Increase maximum motor torque
MAX_TORQUE = 10.0  # Was 5.0 (stronger muscles)
```

**Warning:** These changes trade accuracy for speed. Too much and physics becomes unrealistic.

---

## Part 8: Comparison: Why Our Physics Is Better

**Question:** Why not use simple physics if it's easier?

### Feature Comparison

| Feature | Simple Physics | Our Physics | Why It Matters |
|---------|---|---|---|
| **Gravity** | Always pulls down equally | Pulls less when feet touch ground | Realistic: no fighting gravity when stable |
| **Ground Collision** | "Stop when you hit ground" | Spring + damper bounce | Prevents oscillation, more natural |
| **Joint Speed** | No limit (unrealistic!) | Capped at ±10 rad/s | Prevents jittery movements, more realistic |
| **Touching Ground?** | Yes/No (binary) | Smooth 0.0-1.0 fade | Neural net learns smoother gaits |
| **Body Orientation** | Euler angles (breaks at 90°) | Quaternions (always safe) | Can move freely without getting stuck |
| **Body Rotation** | Treat as angular velocity | Full gyroscopic effects | Spinning bodies actually resist tipping |
| **Weight Distribution** | Single scalar (mass) | 3×3 inertia matrix (per axis) | Different rotation speeds per axis (realistic) |

### Practical Outcomes

```
Simple Physics → Agent learns:
  ✗ Jerky, unrealistic movements
  ✗ Falls over frequently
  ✗ Takes weeks to learn standing
  ✗ Can't use body momentum

Our Physics → Agent learns:
  ✓ Smooth, natural movement
  ✓ Stable stance
  ✓ Learns standing in hours
  ✓ Can use body spin for balance
```

---

## Part 9: Stamina & Energy System

**Why Stamina?**
Real dogs get tired from running. Our agents should too! Stamina adds strategic depth - agents must choose: run fast (tire out) or move efficiently (conserve energy)?

### How Stamina Works

Think of it like a battery:

```
100% stamina → Can move powerfully
 50% stamina → Can move, but less efficiently
 10% stamina → Moving costs more
  0% stamina → EXHAUSTED (big penalty!)

Each step:
- Energetic movement → Stamina depletes faster
- Gentle movement → Stamina depletes slowly
- Standing still → Stamina regenerates
```

### The Numbers

```python
MAX_STAMINA = 100.0              # Full battery
STAMINA_DEPLETION_RATE = 0.5     # Cost per step of hard movement

Example - aggressive running:
- Step 1: Stamina = 100 - 0.5 = 99.5
- Step 2: Stamina = 99.5 - 0.5 = 99.0
- ...200 steps later: Stamina = 0 (exhausted!)

Gentle walking (depletion = 0.1):
- Can continue for 1000 steps before exhaustion

Resting (regeneration = 0.1 per step):
- Recovers 10% stamina per 10 steps
- Full recovery takes ~100 steps
```

### What Stamina Teaches

Agent learns natural strategies:
```
Scenario 1: Reach goal quickly
- Sprint hard (high depletion)
- Arrive in 100 steps with 50% stamina
- But tired for next task

Scenario 2: Reach goal sustainably
- Jog steadily (medium depletion)
- Arrive in 200 steps with 80% stamina
- Ready for next task!

Scenario 3: Balance rest and activity
- Move, then pause to regenerate
- Spreads energy efficiently over time
- Like real animal pacing
```

### Stamina in Agent's Brain

Agent sees stamina level in observations:
```
Observation = [
    joint_angles,           # What angles are my joints?
    joint_velocities,       # How fast are they moving?
    foot_contact,           # Which feet touch ground?
    orientation,            # Which way am I facing?
    com_position,           # Where is my center?
    goal_relative,          # Where is the goal?
    [stamina_fraction]      # How tired am I? (0.0 to 1.0)
]
# Total: 38D observation
```

Agent uses this to make decisions:
- "Low stamina? Rest before attacking."
- "High stamina? Sprint to goal!"
- "Medium stamina? Walk steadily."

---

## Part 10: Movement & Velocity Physics

**Big Picture:** Agent sends motor commands → Physics updates all motion → Agent sees new state → Repeats

### What Happens Each Simulation Step

Think of it as a dance:

```
Step 1: Agent decides
  "Apply torques: [0.5, -0.3, ..., 0.2]"

Step 2: Physics applies those torques to joints
  Each joint starts rotating

Step 3: Joints speed up or slow down
  new_velocity = old_velocity + (torque - friction) * dt

Step 4: Velocity clamped (±10 rad/s max)
  Can't go faster than realistic motors

Step 5: Joints rotate
  new_angle = old_angle + velocity * dt

Step 6: Foot positions computed from joint angles
  Forward kinematics: "where did legs move to?"

Step 7: Check if feet touched ground
  If yes → generate contact

Step 8: Ground pushes back
  Spring-damper force pushes feet upward

Step 9: Body position updates
  Gravity pulls down, contact forces push up
  Net force → acceleration → velocity → position

Step 10: Generate observations
  "Where are joints? Are feet touching? Which way is body tilted?"

Step 11: Calculate reward
  "Did we make progress? Stay stable?"
```

**Time each step takes:** 20 milliseconds (50 steps per second)

### Center of Mass Movement (Linear)

**Simple Physics:**
```
Force applied → Object accelerates
Acceleration → Velocity changes
Velocity → Position changes

Actual math:
a = F / m          (Force divided by mass)
v = v + a*dt       (Velocity updates from acceleration)
x = x + v*dt       (Position updates from velocity)
```

**Real Example - Standing Still:**
```
Quadruped: mass = 5kg, all 4 feet on ground

Forces:
- Gravity: 50 N downward
- Ground pushes: 50 N upward (distributed across 4 feet)
- Net force: 0 N

Result:
- Acceleration: 0 m/s² (no acceleration)
- Velocity: 0 m/s (standing still)
- Position: stays put ✓
```

**Real Example - Jumping:**
```
Agent contracts legs, pushes off ground with extra force

Forces:
- Gravity: 50 N downward
- Ground pushes: 70 N upward
- Net force: 20 N upward

Result:
- Acceleration: 20 / 5 = 4 m/s² upward
- Velocity: increases upward
- Position: body rises ✓
- After leaving ground: gravity brings body back down
```

### Body Rotation (Angular)

**How Bodies Rotate:**
When you push off-center on a spinning top, it spins faster. Same with the quadruped.

```
Torque applied → Rotation accelerates
Angular acceleration → Rotation speed changes
Rotation speed → Orientation changes

Actual math (simplified):
angular_accel = torque / inertia
angular_vel = angular_vel + angular_accel * dt
orientation = orientation + angular_vel * dt  (quaternion)
```

**Why Gyroscopic Effects Matter:**
A spinning top resists tipping. Our agent uses the same physics:
- Running fast while tilted? Momentum helps recovery!
- Spinning body? Harder to fall over!

**Covered in more detail in Part 2**, but key point: Full physics enables more natural recovery strategies.
    return torch.clamp(velocity, -MAX_JOINT_VELOCITY, MAX_JOINT_VELOCITY)
```

**Why this matters:**
- Prevents explosive joint accelerations
- Matches physical motor constraints
- Stabilizes learning by bounding action effects
- Prevents numerical divergence

---

## Part 11: World System - Environment & Goals

**Simple Idea:** Agent is always at the center of its own view. The world moves around it, not the other way around.

### The Agent-Centered World

```
Global view:
- Agent at (87, 45)
- Goal at (92, 50)
- Distance: 7.0 units away

Agent-centered view:
- Agent at (0, 0)  ← Always!
- Goal at (5, 5)  ← Relative to agent
- Distance: 7.0 units away (same!)

Why?
- Simplifies learning (same relative view for all positions)
- Agent learns "move toward goal relative to me" not "move to (92,50)"
- Transfers to different starting positions
```

### Goal System

Each episode has one active goal:

```
Goal spawned at random location in world
Goal stays still (doesn't move)
Agent tries to reach it
When reached (distance < 2 units):
  ✓ Reward bonus
  ✗ Old goal disappears
  ✓ New goal spawns elsewhere
Agent can reach multiple goals in one episode
```

**Real Example - 500 Step Episode:**
```
Step 0: Goal spawned at relative position (10, 8, 2)
Step 100: Agent gets closer (distance = 5 units)
Step 200: Goal reached! (distance < 2) → Goal 1 complete
         New goal spawns at (4, -6, 1)
Step 300: Moving toward new goal...
Step 400: Goal 2 reached!
Step 500: Episode ends
Result: 2 goals reached in one episode!
```

### Initialization - Starting Position

Every episode starts the same way:

```
Agent position: (0, 0, 0.5)  ← 50cm above ground
Orientation: Upright (not tilted)
All joint angles: Ready to stand
All velocities: Zero (standing still)

Then goal is placed somewhere random
Agent has to figure out how to walk to it
```

### World Boundaries

The world has limits. Agent can't go infinitely far:

```
World bounds: X ∈ [-32, 32], Y ∈ [-32, 32], Z ∈ [0, 32]

If agent hits boundary:
- Body is gently clamped (can't exit)
- Small penalty applied (-0.1 per unit penetration)
- Agent learns not to waste time at walls

Example:
- Agent at x=31, tries to move x=+5
- Clamped to x=32 (boundary)
- Gets -0.1 penalty
- Learns: "Don't push on walls, go back to explore"
```

---

## Part 12: How Everything Works Together

**The Big Loop:** Physics → Reward → Learning → Better Physics

### What Happens During Training

```
REPEAT 1000 times (training episodes):

  RESET:
  - Agent standing at origin
  - Goal spawns randomly
  - Stamina = 100

  REPEAT 500 times per episode (timesteps):
    1. Physics engine looks at agent state
       "Where are joints? How is body tilted?"
       
    2. Neural network makes decision
       "Apply these 12 motor torques"
       
    3. Physics updates (20ms simulation):
       - Gravity pulls down
       - Motors push joints
       - Feet detect ground
       - Contact forces push up
       - Body position updates
       - Agent drifts toward goal
       
    4. Physics computes reward (good/bad?)
       "Got closer to goal: +0.2 reward"
       "Tilt too much: -0.5 penalty"
       "Energy cost: -0.1"
       "Total: +0.6 reward for this step"
       
    5. Learning happens (in background)
       "These motor commands got +0.6 reward"
       "Increase probability of this action"
       
    6. If reached goal or took 500 steps
       Episode done, restart

  ANALYZE:
  - Average reward this episode: 50.3
  - Goals reached: 2
  - Final distance to current goal: 3.2
```

### How Rewards Train the Network

```
Neural Network "Brain":
- Input: Joint angles, contact, orientation, goal position
- Output: 12 motor torques (what to do)

Training via PPO:
- If torques led to positive reward → increase their probability
- If torques led to negative reward → decrease their probability
- Over 1000 episodes, brain learns: "these patterns work"

Concrete example:
Action: [0.5, -0.3, 0.1, ...]  (motor commands)
Result: Body walked forward, reached goal
Reward: +0.6
Learning: "When inputs are X, action [0.5, -0.3, 0.1, ...] is good!"

Next time similar inputs appear, brain repeats it
After many episodes, brain specializes in walking
```

### Key Integration Points

1. **Physics → Observations**
   - Joint angles feed back into observations
   - Contact states become inputs
   - Body tilt affects next decision

2. **Observations → Decisions**
   - Neural network sees current state
   - Outputs motor torques
   - Torques change next state

3. **Decisions → Physics**
   - Motors apply torques to joints
   - Physics simulates consequences
   - World state changes

4. **Physics → Reward**
   - Reward evaluates decision quality
   - Positive/negative signal guides learning
   - Network adjusts to maximize future reward

**It's a feedback loop:** Physics → Observations → Decisions → Physics → Reward → Learning → Better Decisions

---

## Part 13: Debugging Common Physics Issues

**When things go wrong, here's how to fix them:**

### Problem: Agent Shakes/Vibrates Uncontrollably

**What it looks like:**
- Joints twitch rapidly
- Body oscillates back and forth
- Learning doesn't improve

**Why it happens:**
- Ground is too bouncy (feet bounce repeatedly)
- Joints are too responsive (react too quickly)
- Physics timestep too large (jumps instead of smooth)

**How to fix it:**
```python
# Option 1: Dampen the ground (less bouncy)
CONTACT_DAMPING = 0.3  # Was 0.2 (absorb more bounce)

# Option 2: Add joint damping (joints resist motion)
JOINT_DAMPING = 0.015  # Was 0.01 (smoother movement)

# Option 3: More frequent physics updates (smoother)
DT = 1/100  # Was 1/50 (update 100 times per second instead of 50)
```

### Problem: Agent Falls Immediately / Can't Stand

**What it looks like:**
- Agent stands for 1 frame then collapses
- Feet don't stick to ground
- Body sinks through floor

**Why it happens:**
- Ground too soft (feet sink in)
- Contact detection failing (feet think they're floating)

**How to fix it:**
```python
# Stiffer ground (pushes feet up more)
CONTACT_STIFFNESS = 0.8  # Was 0.5

# Bigger detection zone (feet recognized from further away)
FOOT_HEIGHT_THRESHOLD = 0.08  # Was 0.05
```

### Problem: Agent Learns Nothing / Reward Always Zero

**What it looks like:**
- Reward chart is flat (all zeros)
- Agent not improving over episodes
- Loss doesn't decrease

**Why it happens:**
- Goal too far/unreachable
- Reward signal too weak
- Agent can't move

**Debug steps:**
```python
# Step 1: Is the agent moving at all?
print(f"Distance traveled: {agent.distance_moved}")
if distance_moved < 0.1:
    print("Agent not moving - increase motor torque")
    MAX_TORQUE = 10.0  # Was 5.0

# Step 2: Is the goal reachable?
print(f"Distance to goal: {agent.distance_to_goal}")
if distance > 30:  # Way too far
    print("Goal spawning too far away")
    spawn_goal_closer()  # Reduce world size or adjust spawning

# Step 3: Are rewards being computed?
print(f"Reward components: goal={reward_goal}, contact={reward_contact}, tilt={reward_tilt}")
if all rewards are 0:
    print("Reward function broken")
    check_reward_computation()
```

### Problem: Agent Can't Get Up / Recover From Falls

**What it looks like:**
- Falls over easily
- Can't stand back up
- Movement sluggish

**Why it happens:**
- Body too heavy (hard to rotate back up)
- Motors too weak (can't apply enough force)
- Max joint speed too slow (can't move fast enough)

**How to fix it:**
```python
# Lighter body (easier to rotate)
I_tensor = [0.3, 0.7, 0.5]  # Was [0.42, 1.04, 0.84]

# Stronger motors
MAX_TORQUE = 8.0  # Was 5.0

# Faster joint movement allowed
MAX_JOINT_VELOCITY = 15.0  # Was 10.0
```

### Problem: Training Very Slow / Takes Forever

**What it looks like:**
- Reward improves but super slowly
- Takes 500+ episodes for any progress
- Learning curves flatten

**Why it happens:**
- Reward signal too sparse (not enough feedback)
- Goal too hard to reach
- Learning rate too small

**How to fix it:**
```python
# Increase learning rate
LR = 3e-4  # Was 1e-4 (learn faster)

# Make goals easier
GOAL_THRESHOLD = 3.0  # Was 2.0 (bigger target)

# Stronger reward signal
GOAL_REWARD_SCALE = 2.0  # Was 1.0 (reward twice as much for reaching goal)
```
```

---

## Part 14: Testing Checklist

**After making physics changes, test these to make sure everything works:**

- [ ] **Can stand**: Agent stands upright with 4 feet touching
- [ ] **Falls realistically**: Loses contact, body falls smoothly
- [ ] **Bounces right**: Feet contact ground without oscillating
- [ ] **Navigation works**: Can move toward goals in different directions
- [ ] **Rewards make sense**: Gets positive rewards for progress
- [ ] **No shaking**: Joints move smoothly, no twitching
- [ ] **Learns quickly**: Training shows improvement in first 10 episodes
- [ ] **Rotations smooth**: Can tilt and recover without gimbal lock
- [ ] **No explosions**: Physics doesn't crash with huge forces
- [ ] **Realistic gait**: Movement looks somewhat natural (not skating)

---

## Part 15: References & Further Reading

For deeper understanding:

- **Rigid Body Physics**: Murray, Sastry, Sastry "A Mathematical Introduction to Robotic Manipulation"
- **Quaternions & 3D Rotations**: Graphics gems, quaternion tutorials
- **Sequential Impulses**: Erin Catto "Box2D" physics engine papers
- **PPO Training**: Schulman et al. "Proximal Policy Optimization Algorithms"
- **Deep Reinforcement Learning**: Sutton & Barto "Reinforcement Learning: An Introduction"

---

## Part 16: Future Enhancements

**Things that could make physics even better:**

1. **Better Friction**: Currently simplified; could add static/kinetic friction distinction
2. **Joint Limits**: Knees shouldn't bend backwards
3. **Muscle Actuators**: More realistic motor models
4. **Optimized Collision**: Faster contact detection with spatial hashing
5. **Deformable Ground**: Soft sand, mud, etc.
6. **Sensor Noise**: Add realism with noisy contact/position readings

These are nice-to-have features for future versions.
