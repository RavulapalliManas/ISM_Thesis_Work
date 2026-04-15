"""Quick diagnostic — run directly: python project5_symmetry/_test_arena.py"""
import warnings; warnings.filterwarnings('ignore')
import sys, time

print("1. importing gym..."); sys.stdout.flush()
import minigrid 

print("2. importing gym_minigrid..."); sys.stdout.flush()
from gym_minigrid.minigrid import MiniGridEnv, Grid, Floor, Wall, MissionSpace
from gym_minigrid.wrappers import RGBImgPartialObsWrapper
import numpy as np

print("3. creating MissionSpace..."); sys.stdout.flush()
ms = MissionSpace(mission_func=lambda: "explore")
print("   ok:", ms); sys.stdout.flush()

print("4. building Grid manually..."); sys.stdout.flush()
g = Grid(10, 10)
g.wall_rect(0, 0, 10, 10)
for c in range(1, 9):
    for r in range(1, 9):
        g.set(c, r, Floor('grey'))
print("   ok"); sys.stdout.flush()

print("5. calling MiniGridEnv.__init__ (subclass)..."); sys.stdout.flush()

class MinimalEnv(MiniGridEnv):
    def __init__(self):
        super().__init__(
            mission_space=MissionSpace(mission_func=lambda: "explore"),
            width=10, height=10, max_steps=1000, agent_view_size=7,
        )
    def _gen_grid(self, w, h):
        print("   _gen_grid called"); sys.stdout.flush()
        self.grid = Grid(w, h)
        self.grid.wall_rect(0, 0, w, h)
        for c in range(1, w-1):
            for r in range(1, h-1):
                self.grid.set(c, r, Floor('grey'))
        print("   calling place_agent..."); sys.stdout.flush()
        self.place_agent()
        print("   place_agent done"); sys.stdout.flush()

t0 = time.time()
e = MinimalEnv()
print(f"   __init__ done in {time.time()-t0:.2f}s"); sys.stdout.flush()

print("6. wrapping with RGBImgPartialObsWrapper..."); sys.stdout.flush()
we = RGBImgPartialObsWrapper(e, tile_size=1)
print("   ok"); sys.stdout.flush()

print("7. calling reset..."); sys.stdout.flush()
obs, info = we.reset()
print("   obs image shape:", obs['image'].shape); sys.stdout.flush()

print("8. calling get_frame..."); sys.stdout.flush()
frame = e.get_frame(highlight=False, tile_size=10)
print("   frame shape:", frame.shape); sys.stdout.flush()

print("\nALL STEPS PASSED")
