import sys
import random
import itertools

import numpy as np
from gym.spaces import Discrete
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import pygame as pg
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE


class MoveBall():
    def __init__(self, settings):
        self.sets = settings
        self.W = settings['W']
        self.H = settings['H']
        self.dt = settings['dt']
        self.action_space, self.acts = self.setup_action_space()
        self.space = None
        self.ply = None
        self.goal = None

    def reset(self):
        self.space = self.setup_space()
        self.ply, self.ply_ctrl = self.spawn_player()
        self.goal = self.spawn_goal()
        self.enemies = self.spawn_enemies(self.sets['n_enemy'])
        return

    def step(self, action):
        v, rot = self.acts[action]
        turn = random.uniform(-100, 100)
        v = random.uniform(0, 100)
        self.ply_ctrl.angle = self.ply.angle - turn
        dv = Vec2d(v * 1.0, 0.0)
        self.ply_ctrl.velocity = self.ply.rotation_vector.cpvrotate(dv)
        self.space.step(self.dt)

    def setup_action_space(self):
        nv, vmin, vmax = self.sets['action_v']
        nr, rmin, rmax = self.sets['action_r']
        action_space = Discrete(nv*nr)
        vsets = np.linspace(vmin, vmax, nv)
        rsets = np.linspace(rmin, rmax, nr)
        acts = list(itertools.product(vsets, rsets))
        return action_space, acts

    def setup_space(self):
        space = pymunk.Space()
        space.gravity = self.sets['gravity']
        space.damping = self.sets['damping']
        # Add walls for surrounding the simulation field.
        b0 = space.static_body
        pts = [(10, 10), (self.W-10, 10), (self.W-10, self.H-10),
               (10, self.H-10)]
        wall_width = 3.0
        for i in range(4):
            seg = pymunk.Segment(b0, pts[i], pts[(i+1) % 4], wall_width)
            seg.elasticity = 1.0
            seg.friction = 0.0
            space.add(seg)
        return space

    def setup_pygame(self):
        pg.init()
        scr = pg.display.set_mode((self.W, self.H))
        clk = pg.time.Clock()
        draw_options = pymunk.pygame_util.DrawOptions(scr)
        return scr, clk, draw_options

    def add_ball(self, r, mass):
        body = pymunk.Body()
        body.position = Vec2d(random.random() * (self.W - r*5),
                              random.random() * (self.H - r*5))
        body.elasticity = 1.0
        # shape = pymunk.Poly.create_box(body, (size, size), 0.0)
        shape = pymunk.Circle(body, r)
        shape.mass = mass
        shape.elasticity = 1.0
        shape.friction = 0.0
        self.space.add(body, shape)
        return body

    def spawn_goal(self):
        s = self.sets['GOAL_R']
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = Vec2d(random.random() * (self.W - s*5),
                              random.random() * (self.H - s*5))
        shape = pymunk.Poly.create_box(body, (s, s), 0.0)
        shape.elasticity = 1.0
        shape.friction = 0.0
        shape.color = (255, 50, 50, 50)
        self.space.add(body, shape)
        return body

    def spawn_player(self):
        px, py = self.W//2, self.H//2
        ply_ctrl = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        ply_ctrl.position = px, py
        self.space.add(ply_ctrl)
        ply = self.add_ball(self.sets['OBJ_R'], 10)
        ply.position = px, py
        for s in ply.shapes:
            s.color = (0, 255, 100, 255)

        pivot = pymunk.PivotJoint(ply_ctrl, ply, (0, 0), (0, 0))
        self.space.add(pivot)
        pivot.max_bias = 0  # disable joint correction
        pivot.max_force = 10000  # emulate linear friction

        gear = pymunk.GearJoint(ply_ctrl, ply, 0.0, 1.0)
        self.space.add(gear)
        gear.error_bias = 0  # attempt to fully correct the joint each step
        gear.max_bias = 1.2  # but limit it's angular correction rate
        gear.max_force = 50000  # emulate angular friction
        return ply, ply_ctrl

    def spawn_enemies(self, n_enemy):
        enemies = []
        R = self.sets['OBJ_R']
        FMAX = 5000
        for _ in range(n_enemy):
            b = self.add_ball(R, 10)
            fx = random.uniform(-FMAX, FMAX)
            fy = random.uniform(-FMAX, FMAX)
            b.apply_impulse_at_local_point((fx, fy))
            enemies.append(b)
        return enemies


if __name__ == '__main__':
    settings = {
        # Parameters for Pymunk
        'W': 500,  # Width of simulation field.
        'H': 500,  # Height of simulation field.
        'gravity': [0, 0],
        'damping': 1.0,
        'dt': 1/30,
        'GOAL_R': 20,
        'OBJ_R': 10,
        'n_enemy': 10,
        # Parameters for Gym APIs
        'action_v': (5, 0.0, 100),  # n-action, min, max for velocity.
        'action_r': (5, -20, 20),   # n-action, min, max for rotation.
    }
    fps = 1/settings['dt']
    env = MoveBall(settings)
    env.reset()
    scr, clk, draw_options = env.setup_pygame()
    while True:
        for e in pg.event.get():
            if e.type == QUIT:
                sys.exit(0)
            elif e.type == KEYDOWN and e.key == K_ESCAPE:
                sys.exit(0)
        scr.fill(pg.Color("black"))
        env.space.debug_draw(draw_options)
        a = env.action_space.sample()
        env.step(a)
        pg.display.flip()
        clk.tick(fps)
    pg.quit()
