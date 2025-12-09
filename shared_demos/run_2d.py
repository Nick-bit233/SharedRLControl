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
DT = 0.05

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
    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("2D Shared Control Demo")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont(None, 24)

    # Initialize Agent
    try:
        print(f"Initializing agent on {args.device}...")
        agent = Agent(args.checkpoint, device=args.device)
        print("Agent initialized.")
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        import traceback
        traceback.print_exc()
        pygame.quit()
        return

    # Simulation State
    drone_pos = np.array([0.0, 0.0])
    drone_vel = np.array([0.0, 0.0])
    drone_yaw = 0.0  # heading angle (rad)
    prev_action = np.zeros(4) # [vx, vy, vz, yaw_rate] (body-frame)

    # Trajectory history for drawing
    trajectory = []

    running = True
    try:
        while running:
            # 1. Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # 2. Get User Input - joystick style
            # W/S : throttle (forward/back) along body x axis
            # A/D : yaw rate (turn)
            keys = pygame.key.get_pressed()
            throttle = 0.0
            yaw_input = 0.0

            max_throttle = 2.0  # m/s forward
            max_yaw_rate = 1.5  # rad/s

            if keys[pygame.K_w]: throttle += 1.0
            if keys[pygame.K_s]: throttle -= 1.0
            if keys[pygame.K_a]: yaw_input -= 1.0
            if keys[pygame.K_d]: yaw_input += 1.0

            # Human action in body frame: forward throttle along x, lateral y set to 0
            human_vx_body = float(throttle) * max_throttle
            human_vy_body = 0.0
            human_yaw_rate = float(yaw_input) * max_yaw_rate

            human_action = np.array([human_vx_body, human_vy_body, 0.0, human_yaw_rate])

            # 3. Prepare Model Input
            # State: [vel_b(3), ang_vel_b(3), orientation(4)]
            # Assuming identity orientation [1, 0, 0, 0] (w, x, y, z)
            # So vel_b = vel_w
            
            # Construct state tensor
            # vel_b (3) = world vel rotated into body frame: v_b = R(-yaw) * v_w
            cy = np.cos(drone_yaw)
            sy = np.sin(drone_yaw)
            vx_w, vy_w = float(drone_vel[0]), float(drone_vel[1])
            vel_b_x = cy * vx_w + sy * vy_w
            vel_b_y = -sy * vx_w + cy * vy_w
            vel_b = np.array([vel_b_x, vel_b_y, 0.0])

            # ang_vel_b (3) - body angular velocity about z is current yaw rate (approx)
            # we don't model angular inertia here, approximate with previous action yaw rate
            ang_vel_b = np.array([0.0, 0.0, prev_action[3]])

            # orientation (4) - quaternion from yaw: (w, x, y, z)
            qw = np.cos(drone_yaw * 0.5)
            qz = np.sin(drone_yaw * 0.5)
            orientation = np.array([qw, 0.0, 0.0, qz])

            drone_state_np = np.concatenate([vel_b, ang_vel_b, orientation])
            
            # Convert to tensors
            drone_state_t = torch.tensor(drone_state_np, dtype=torch.float32, device=args.device)
            human_action_t = torch.tensor(human_action, dtype=torch.float32, device=args.device)
            prev_action_t = torch.tensor(prev_action, dtype=torch.float32, device=args.device)

            # 4. Agent Inference
            action = agent.act(drone_state_t, human_action_t, prev_action_t)
            # action is [vx, vy, vz, yaw_rate]

            # 5. Update Simulation
            # Agent action is body-frame desired velocities and yaw_rate.
            # Convert body-frame command to world-frame using current yaw, then integrate.
            cmd_vel_body = np.array([float(action[0]), float(action[1])])
            cy = np.cos(drone_yaw)
            sy = np.sin(drone_yaw)
            # body -> world: v_w = R(yaw) * v_b
            cmd_vel_world = np.array([cy * cmd_vel_body[0] - sy * cmd_vel_body[1],
                                      sy * cmd_vel_body[0] + cy * cmd_vel_body[1]])

            drone_vel = 0.1 * drone_vel + 0.9 * cmd_vel_world # Simple smoothing
            drone_pos += drone_vel * DT

            # Update yaw from agent yaw_rate output
            yaw_rate = float(action[3])
            drone_yaw += yaw_rate * DT

            # Boundary check
            drone_pos = np.clip(drone_pos, -MAP_SIZE, MAP_SIZE)

            # Store trajectory
            trajectory.append(drone_pos.copy())
            if len(trajectory) > 500: # Limit history
                trajectory.pop(0)

            # Update prev_action
            prev_action = np.array([float(action[0]), float(action[1]), float(action[2]), float(action[3])])

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

            # Draw Trajectory
            if len(trajectory) > 1:
                points = [world_to_screen(p) for p in trajectory]
                pygame.draw.lines(screen, GREEN, False, points, 2)

            # Draw Drone
            drone_screen_pos = world_to_screen(drone_pos)
            pygame.draw.circle(screen, BLUE, drone_screen_pos, 10)
            
            # Draw Heading Indicator (small black line)
            heading_len = 15.0
            heading_end = (
                drone_screen_pos[0] + heading_len * np.cos(drone_yaw),
                drone_screen_pos[1] - heading_len * np.sin(drone_yaw) # screen y is inverted
            )
            pygame.draw.line(screen, BLACK, drone_screen_pos, heading_end, 2)

            # Draw Human Input (Green) - Rotate body frame input to world frame for display
            # human_action is [vx_b, vy_b, vz_b, yaw_rate]
            h_vx_b, h_vy_b = human_action[0], human_action[1]
            h_vx_w = np.cos(drone_yaw) * h_vx_b - np.sin(drone_yaw) * h_vy_b
            h_vy_w = np.sin(drone_yaw) * h_vx_b + np.cos(drone_yaw) * h_vy_b
            draw_arrow(screen, drone_pos, np.array([h_vx_w, h_vy_w]), GREEN, scale=1.0)

            # Draw Agent Output (Red) - Rotate body frame output to world frame for display
            # action is [vx_b, vy_b, vz_b, yaw_rate]
            a_vx_b, a_vy_b = action[0], action[1]
            a_vx_w = np.cos(drone_yaw) * a_vx_b - np.sin(drone_yaw) * a_vy_b
            a_vy_w = np.sin(drone_yaw) * a_vx_b + np.cos(drone_yaw) * a_vy_b
            draw_arrow(screen, drone_pos, np.array([a_vx_w, a_vy_w]), RED, scale=1.0)

            # Info Text
            text = font.render(f"Human: [{human_action[0]:.2f}, {human_action[1]:.2f}, yaw: {human_action[3]:.2f}]", True, BLACK)
            screen.blit(text, (10, 10))
            text = font.render(f"Agent: [{action[0]:.2f}, {action[1]:.2f}, yaw: {action[3]:.2f}]", True, BLACK)
            screen.blit(text, (10, 30))

            pygame.display.flip()
            clock.tick(int(1/DT))
    except Exception as e:
        print(f"Simulation crashed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()
