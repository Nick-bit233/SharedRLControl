import pygame
import torch

class JoystickInterface:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        
        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"已连接手柄: {self.joystick.get_name()}")
            self.connected = True
        else:
            print("未检测到手柄，将使用默认零输入。")
            self.connected = False

    def get_input(self):
        """
        返回: torch.Tensor [roll, pitch, throttle, yaw_rate]
        范围通常是 -1 到 1
        注意：不同手柄的轴映射可能不同 (Axis 0-3)，请根据实际情况调整索引
        """
        pygame.event.pump() # 处理事件队列
        
        if not self.joystick:
            return torch.zeros(4)

        # 示例映射 (Xbox Controller):
        # Axis 0: 左摇杆 X (左右) -> Yaw Rate
        # Axis 1: 左摇杆 Y (上下) -> Throttle (通常需要反向)
        # Axis 2: 右摇杆 X (左右) -> Roll (左右横移)
        # Axis 3: 右摇杆 Y (上下) -> Pitch (前后移动)
        
        # 读取原始值
        yaw_rate = self.joystick.get_axis(0)
        throttle = -self.joystick.get_axis(1) # 上推通常是负值，取反
        roll = self.joystick.get_axis(2)
        pitch = -self.joystick.get_axis(3)    # 上推通常是负值，取反

        # 死区处理 (防止漂移)
        deadzone = 0.1
        inputs = [roll, pitch, throttle, yaw_rate]
        inputs = [0.0 if abs(x) < deadzone else x for x in inputs]
        
        return torch.tensor(inputs, dtype=torch.float32)