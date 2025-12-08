import pygame
import torch
import numpy as np
import argparse
import sys
import os

# Add agent to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from agent import Agent

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
MAP_SIZE = 20.0  # meters (half-size, so -20 to 20)
PIXELS_PER_METER = SCREEN_WIDTH / (2 * MAP_SIZE)
DT = 0.05  # 20 Hz

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GRAY = (200, 200, 200)

def world_to_screen(pos):
    # pos: (x, y) in meters
    # screen: (u, v) in pixels
    # x: -20 -> 0, 20 -> 800.  u = (x + 20) * scale
    # y: -20 -> 800, 20 -> 0.  v = (20 - y) * scale
    u = int((pos[0] + MAP_SIZE) * PIXELS_PER_METER)
    v = int((MAP_SIZE - pos[1]) * PIXELS_PER_METER)
    return (u, v)

def draw_arrow(screen, start_pos, vector, color, scale=1.0):
    # vector is velocity in m/s
    end_pos_world = start_pos + vector * scale
    start_screen = world_to_screen(start_pos)
    end_screen = world_to_screen(end_pos_world)
    pygame.draw.line(screen, color, start_screen, end_screen, 3)
    # Draw simple arrow head
    # ... (omitted for simplicity, line is enough for now)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    args = parser.parse_args()

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("NavRL 2D Shared Control Demo")
    clock = pygame.time.Clock()

    # Initialize Agent
    print(f"Initializing agent on {args.device}...")
    agent = Agent(args.checkpoint, device=args.device)
    print("Agent initialized.")

    # Simulation State
    drone_pos = np.array([0.0, 0.0])
    drone_vel = np.array([0.0, 0.0])
    prev_action = np.zeros(4) # [vx, vy, vz, yaw_rate]

    running = True
    while running:
        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 2. Get User Input
        keys = pygame.key.get_pressed()
        human_vx = 0.0
        human_vy = 0.0
        
        speed = 1.0 # m/s
        if keys[pygame.K_w]: human_vy += speed
        if keys[pygame.K_s]: human_vy -= speed
        if keys[pygame.K_a]: human_vx -= speed
        if keys[pygame.K_d]: human_vx += speed
        
        # Normalize if diagonal
        if human_vx != 0 and human_vy != 0:
            human_vx *= 0.707
            human_vy *= 0.707

        human_action = np.array([human_vx, human_vy, 0.0, 0.0]) # [vx, vy, vz, yaw_rate]

        # 3. Prepare Model Input
        # State: [vel_b(3), ang_vel_b(3), orientation(4)]
        # Assuming identity orientation [1, 0, 0, 0] (w, x, y, z)
        # So vel_b = vel_w
        
        # Construct state tensor
        # vel_b (3)
        vel_b = np.array([drone_vel[0], drone_vel[1], 0.0])
        # ang_vel_b (3)
        ang_vel_b = np.array([0.0, 0.0, 0.0])
        # orientation (4) - w, x, y, z
        orientation = np.array([1.0, 0.0, 0.0, 0.0]) # Identity
        
        drone_state_np = np.concatenate([vel_b, ang_vel_b, orientation])
        
        # Convert to tensors
        drone_state_t = torch.tensor(drone_state_np, dtype=torch.float32, device=args.device)
        human_action_t = torch.tensor(human_action, dtype=torch.float32, device=args.device)
        prev_action_t = torch.tensor(prev_action, dtype=torch.float32, device=args.device)

        # 4. Agent Inference
        action = agent.act(drone_state_t, human_action_t, prev_action_t)
        # action is [vx, vy, vz, yaw_rate]

        # 5. Update Simulation
        # Simple kinematics: pos += vel * dt
        # Assume the drone perfectly tracks the commanded velocity (simplified)
        # Or use simple inertia: vel = 0.8 * vel + 0.2 * command
        
        cmd_vel = action[:2]
        drone_vel = 0.1 * drone_vel + 0.9 * cmd_vel # Simple smoothing
        
        drone_pos += drone_vel * DT
        
        # Boundary check
        drone_pos = np.clip(drone_pos, -MAP_SIZE, MAP_SIZE)

        # Update prev_action
        prev_action = action

        # 6. Rendering
        screen.fill(WHITE)

        # Draw Grid
        for i in range(-int(MAP_SIZE), int(MAP_SIZE) + 1, 5):
            # Vertical lines
            start = world_to_screen((i, -MAP_SIZE))
            end = world_to_screen((i, MAP_SIZE))
            pygame.draw.line(screen, GRAY, start, end, 1)
            # Horizontal lines
            start = world_to_screen((-MAP_SIZE, i))
            end = world_to_screen((MAP_SIZE, i))
            pygame.draw.line(screen, GRAY, start, end, 1)

        # Draw Drone
        drone_screen_pos = world_to_screen(drone_pos)
        pygame.draw.circle(screen, BLUE, drone_screen_pos, 10)

        # Draw Human Input (Green)
        draw_arrow(screen, drone_pos, human_action[:2], GREEN, scale=1.0)

        # Draw Agent Output (Red)
        draw_arrow(screen, drone_pos, action[:2], RED, scale=1.0)

        # Info Text
        font = pygame.font.SysFont(None, 24)
        text = font.render(f"Human: [{human_action[0]:.2f}, {human_action[1]:.2f}]", True, BLACK)
        screen.blit(text, (10, 10))
        text = font.render(f"Agent: [{action[0]:.2f}, {action[1]:.2f}]", True, BLACK)
        screen.blit(text, (10, 30))

        pygame.display.flip()
        clock.tick(int(1/DT))

    pygame.quit()

if __name__ == "__main__":
    main()
