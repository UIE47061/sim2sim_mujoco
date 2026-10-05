import multiprocessing as mp
import pygame
import numpy as np
import time

def pygame_worker(q, vx_scale, vy_scale, yaw_scale):
    pygame.init()
    screen = pygame.display.set_mode((400, 200))
    pygame.display.set_caption("Keyboard Control Instructions")
    font = pygame.font.SysFont("Arial", 18)
    
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        screen.fill((20, 20, 20))
        lines = [
            "Keyboard Control Rules:",
            "-----------------------------",
            "W / S : Move Forward / Backward (vx)",
            "A / D : Move Left / Right     (vy)",
            "Q / E : Rotate Left / Right   (yaw)",
        ]
        y = 20
        for line in lines:
            text_surf = font.render(line, True, (255, 255, 255))
            screen.blit(text_surf, (20, y))
            y += 30
        pygame.display.flip()

        keys = pygame.key.get_pressed()
        target_vx, target_vy, target_yaw = 0.0, 0.0, 0.0

        if keys[pygame.K_w]: target_vx = vx_scale
        elif keys[pygame.K_s]: target_vx = -vx_scale
        
        if keys[pygame.K_a]: target_vy = vy_scale
        elif keys[pygame.K_d]: target_vy = -vy_scale
        
        if keys[pygame.K_q]: target_yaw = yaw_scale
        elif keys[pygame.K_e]: target_yaw = -yaw_scale
        
        q.put((target_vx, target_vy, target_yaw))
        
        clock.tick(50)

    pygame.quit()

class KeyboardController:
    def __init__(self, vx_scale=1.0, vy_scale=1.0, yaw_scale=1.0, smooth=0.2):
        self.vx_scale = vx_scale
        self.vy_scale = vy_scale
        self.yaw_scale = yaw_scale
        self.smooth = smooth
        
        self.vx = 0.0
        self.vy = 0.0
        self.yaw = 0.0
        
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_yaw = 0.0
        
        mp.set_start_method('spawn', force=True)
        self.q = mp.Queue()
        self.p = mp.Process(target=pygame_worker, daemon=True, args=(self.q, self.vx_scale, self.vy_scale, self.yaw_scale))
        self.p.start()
        
    def smooth_update(self, old, new):
        return old * (1 - self.smooth) + new * self.smooth

    def read(self):
        while not self.q.empty():
            self.target_vx, self.target_vy, self.target_yaw = self.q.get_nowait()
            
        self.vx = self.smooth_update(self.vx, self.target_vx)
        self.vy = self.smooth_update(self.vy, self.target_vy)
        self.yaw = self.smooth_update(self.yaw, self.target_yaw)

        return np.array([self.vx, self.vy, self.yaw], dtype=np.float32)
