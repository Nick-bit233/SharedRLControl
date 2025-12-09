import pygame
import numpy as np
import sys
import os

# # Force X11 driver if not set, to avoid some Wayland/GL issues in remote envs
# if "SDL_VIDEODRIVER" not in os.environ:
#     os.environ["SDL_VIDEODRIVER"] = "x11"

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
MAP_SIZE = 20.0  # meters (half-size, so -20 to 20)
MAX_SPEED = 5.0  # m/s
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
    u = int((pos[0] + MAP_SIZE) * PIXELS_PER_METER)
    v = int((MAP_SIZE - pos[1]) * PIXELS_PER_METER)
    return (u, v)

def main():
    print("Initializing Pygame...")
    pygame.display.init()
    pygame.font.init()
    print("Pygame initialized.")
    
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Drone Control 2D Minimal Demo")
    clock = pygame.time.Clock()

    font = pygame.font.Font(None, 24)

    # Simulation State
    drone_pos = np.array([0.0, 0.0])
    drone_vel = np.array([0.0, 0.0])
    
    # Trajectory history for drawing
    trajectory = []

    print("Starting main loop...")
    running = True
    try:
        while running:
            # 1. Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # 2. Get User Input
            keys = pygame.key.get_pressed()
            target_vx = 0.0
            target_vy = 0.0
            
            speed = MAX_SPEED # m/s
            if keys[pygame.K_w]: target_vy += speed
            if keys[pygame.K_s]: target_vy -= speed
            if keys[pygame.K_a]: target_vx -= speed
            if keys[pygame.K_d]: target_vx += speed
            
            # Normalize if diagonal
            if target_vx != 0 and target_vy != 0:
                target_vx *= 0.707
                target_vy *= 0.707

            # 3. Update Simulation
            # Simple smoothing
            drone_vel = 0.1 * drone_vel + 0.9 * np.array([target_vx, target_vy])
            drone_pos += drone_vel * DT
            
            # Boundary check
            drone_pos = np.clip(drone_pos, -MAP_SIZE, MAP_SIZE)
            
            # Store trajectory
            trajectory.append(drone_pos.copy())
            if len(trajectory) > 500: # Limit history
                trajectory.pop(0)

            # 4. Rendering
            screen.fill(WHITE)

            # Draw Grid
            for i in range(-int(MAP_SIZE), int(MAP_SIZE) + 1, 5):
                start = world_to_screen((i, -MAP_SIZE))
                end = world_to_screen((i, MAP_SIZE))
                pygame.draw.line(screen, GRAY, start, end, 1)
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

            # Info Text
            text = font.render(f"Pos: [{drone_pos[0]:.2f}, {drone_pos[1]:.2f}]", True, BLACK)
            screen.blit(text, (10, 10))
            text = font.render(f"Vel: [{drone_vel[0]:.2f}, {drone_vel[1]:.2f}]", True, BLACK)
            screen.blit(text, (10, 30))

            pygame.display.flip()
            clock.tick(int(1/DT))
            
    except Exception as e:
        print(f"Error in main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Quitting...")
        pygame.quit()

if __name__ == "__main__":
    main()
