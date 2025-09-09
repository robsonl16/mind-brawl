import gymnasium as gym
import numpy as np
import pygame
import time
from pylsl import StreamInlet, resolve_byprop

# === Set thresholds and gains ===
STEER_THRESHOLD = 30.0     # degrees/sec
STEER_GAIN = 1.0           # steering output scaling
ACCEL_THRESHOLD = 10.0
BRAKE_THRESHOLD = 70.0
ACCEL_GAIN = 1.0
BRAKE_GAIN = 0.8

# === Connect to Muse gyro stream ===
print("Searching for Muse gyroscope LSL stream...")
streams = resolve_byprop('type', 'GYRO', timeout=15.0)
inlet = StreamInlet(streams[0])
print("Connected to Muse gyroscope!")

# === Set up pygame to allow ESC key ===
pygame.init()
screen = pygame.display.set_mode((100, 100))  # dummy window

# === Create the CarRacing environment ===
env = gym.make("CarRacing-v3", render_mode="human", continuous=True)
obs, _ = env.reset()

clock = pygame.time.Clock()

print("Use head tilts to steer (←/→) and nods to accelerate (↑) or brake (↓). ESC to quit.")

prev_steer = 0.0  # Initialize previous steer value
prev_gas = 0.0    # Initialize previous gas value

# === Main loop ===
for step in range(5000):
    # === Get gyro data ===
    latest_sample = [0.0, 0.0, 0.0]
    while True:
        sample, timestamp = inlet.pull_sample(timeout=0.001)
        if sample is None:
            break
        latest_sample = sample

    gyro_x, gyro_y, _ = latest_sample


    # === Map gyro to steering ===
    if gyro_x < -STEER_THRESHOLD:
        steer = 1.0  # turn right
    elif gyro_x > STEER_THRESHOLD:
        steer = -1.0 # turn left
    else:
        steer = prev_steer * 0.999  # center if no head movement

    # === Map gyro to throttle and brake ===
    if gyro_y > ACCEL_THRESHOLD:  # nod forward
        gas = np.clip(gyro_y / 10.0, 0, 1.0) * ACCEL_GAIN
        brake = 0.0
    elif gyro_y < -BRAKE_THRESHOLD:  # nod backward
        gas = 0.0
        brake = np.clip(-gyro_y / 90.0, 0, 1.0) * BRAKE_GAIN
    else:
        gas = prev_gas * 0.99
        brake = 0.0

    # === Check for ESC or quit ===
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (
            event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
        ):
            print("Exiting.")
            pygame.quit()
            env.close()
            exit()

    # === Apply to simulation ===
    action = np.array([steer, gas, brake])
    obs, reward, terminated, truncated, _ = env.step(action)

    print(f"Step {step:04d} | Gyro x={gyro_x:+.2f} y={gyro_y:+.2f} → steer={steer:+.2f} gas={gas:.2f} brake={brake:.2f}")

    if terminated or truncated:
        print("Track ended. Resetting.")
        obs, _ = env.reset()

    clock.tick(60)

env.close()
pygame.quit()
