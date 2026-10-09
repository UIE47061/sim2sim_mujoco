"""Held velocity keys and persistent WTW gait/body commands."""

import multiprocessing as mp
from queue import Empty, Full

import numpy as np

from wtw_commands import WTWCommandState


def publish_latest(queue, value):
    try:
        queue.put_nowait(value)
    except Full:
        try:
            queue.get_nowait()
        except Empty:
            pass
        try:
            queue.put_nowait(value)
        except Full:
            pass


def pygame_worker(
    q,
    vx_scale,
    vy_scale,
    yaw_scale,
    wtw_commands=None,
    wtw_limits=None,
    backward_scale=None,
):
    import pygame

    pygame.init()
    state = (
        WTWCommandState(wtw_commands, wtw_limits) if wtw_commands is not None else None
    )
    screen = pygame.display.set_mode((690, 590) if state else (480, 230))
    pygame.display.set_caption("WTW Keyboard Control" if state else "Keyboard Control")
    font = pygame.font.SysFont("Arial", 18)
    pygame.key.set_repeat(250, 100)
    clock = pygame.time.Clock()
    running = True
    backward_scale = vx_scale if backward_scale is None else backward_scale
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif state:
                    state.handle(pygame.key.name(event.key))
        keys = pygame.key.get_pressed()
        velocity = np.array(
            [
                vx_scale
                if keys[pygame.K_w]
                else -backward_scale
                if keys[pygame.K_s]
                else 0.0,
                vy_scale
                if keys[pygame.K_a]
                else -vy_scale
                if keys[pygame.K_d]
                else 0.0,
                yaw_scale
                if keys[pygame.K_q]
                else -yaw_scale
                if keys[pygame.K_e]
                else 0.0,
            ],
            dtype=np.float32,
        )
        if not running or keys[pygame.K_SPACE]:
            velocity[:] = 0
        if state:
            state.command[:3] = velocity
            publish_latest(q, state.command.tolist())
        else:
            publish_latest(q, velocity.tolist())
        screen.fill((20, 20, 20))
        lines = [
            "Click this window to control the robot.",
            "W / S : forward / backward (release to stop)",
            "A / D : left / right; Q / E : turn left / right",
            "Space : stop velocity; Esc : close keyboard window",
        ]
        if state:
            c = state.command
            lines += [
                "1 trot | 2 pace | 3 bound | 4 pronk",
                "R / F : frequency +/-     T / G : body height +/-",
                "Y / H : swing height +/-  U / J : pitch +/-",
                "I / K : roll +/-          O / L : stance width +/-",
                "P / ; : stance length +/-  ] / [ : duty factor +/-",
                "0 : restore default gait and body parameters",
                "Gait switches preserve phase and observation history.",
                "",
                f"Gait: {state.gait} | vx {c[0]:+.2f} vy {c[1]:+.2f} yaw {c[2]:+.2f}",
                f"Frequency {c[4]:.2f} Hz | duty {c[8]:.3f}",
                f"Height offset {c[3]:+.3f} m | swing {c[9]:.3f} m",
                f"Pitch {c[10]:+.3f} rad | roll {c[11]:+.3f} rad",
                f"Stance width {c[12]:.3f} m | length {c[13]:.3f} m",
                "Swing/body values are commands; actual tracking can differ.",
            ]
        for index, line in enumerate(lines):
            screen.blit(font.render(line, True, (235, 235, 235)), (18, 15 + index * 30))
        pygame.display.flip()
        clock.tick(50)
    pygame.quit()


class KeyboardController:
    def __init__(
        self,
        vx_scale=1.0,
        vy_scale=1.0,
        yaw_scale=1.0,
        smooth=0.2,
        wtw_commands=None,
        wtw_limits=None,
        backward_scale=None,
    ):
        if not 0 < smooth <= 1:
            raise ValueError("Keyboard smoothing must be in (0, 1]")
        self.vx_scale, self.vy_scale, self.yaw_scale = vx_scale, vy_scale, yaw_scale
        self.backward_scale = vx_scale if backward_scale is None else backward_scale
        self.smooth = smooth
        self.command = (
            WTWCommandState(wtw_commands, wtw_limits).command
            if wtw_commands is not None
            else np.zeros(3, dtype=np.float32)
        )
        self.target = self.command.copy()
        context = mp.get_context("spawn")
        self.q = context.Queue(maxsize=1)
        self.p = context.Process(
            target=pygame_worker,
            daemon=True,
            args=(
                self.q,
                vx_scale,
                vy_scale,
                yaw_scale,
                wtw_commands,
                wtw_limits,
                self.backward_scale,
            ),
        )
        self.p.start()

    def smooth_update(self, old, new):
        return old * (1 - self.smooth) + new * self.smooth

    def read(self):
        while True:
            try:
                self.target[:] = self.q.get_nowait()
            except Empty:
                break
        if not self.p.is_alive():
            self.target[:3] = 0
        self.command[:3] = self.smooth_update(self.command[:3], self.target[:3])
        self.command[3:] = self.target[3:]
        return self.command.copy()

    def close(self):
        if self.p.is_alive():
            self.p.terminate()
        self.p.join(timeout=2)
        self.q.close()
