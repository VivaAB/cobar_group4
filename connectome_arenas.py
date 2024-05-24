from flygym.arena import FlatTerrain, BaseArena
import numpy as np

class OptomotorTerrain(FlatTerrain):
    def __init__(
        self,
        n=18,
        height=100,
        distance=12,
        ang_speed=1,
        light=True,
        dark=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.height = height
        self.ang_speed = ang_speed

        self.cylinders = []
        self.phase = 0
        self.curr_time = 0
        
        if not dark:
            palette1=(0.5,0.5,0.5,1)
        else:
            palette1=(0,0,0,1)
        
        if not light:
            palette2=(0.5,0.5,0.5,1)
        else:
            palette2= (1,1,1,1)
            
        palette = (palette1, palette2)

        cylinder_material = self.root_element.asset.add(
            "material", name="cylinder", reflectance=0.1
        )
        init_pos = np.exp(2j * np.pi * np.arange(n) / n) * distance
        radius = np.abs(init_pos[1] - init_pos[0]) / 2
        self.init_pos = init_pos

        for i, pos in enumerate(self.init_pos):
            cylinder = self.root_element.worldbody.add(
                "body",
                name=f"cylinder_{i}",
                mocap=True,
                pos=(pos.real, pos.imag, self.height / 2),
            )

            cylinder.add(
                "geom",
                type="cylinder",
                size=(radius, self.height / 2),
                rgba=palette[i % len(palette)],
                material=cylinder_material,
            )

            self.cylinders.append(cylinder)

        self.birdeye_cam = self.root_element.worldbody.add(
            "camera",
            name="birdeye_cam",
            mode="fixed",
            pos=(0, 0, 25),
            euler=(0, 0, 0),
            fovy=45,
        )

    def reset(self, physics):
        """Resets the position of the cylinders and the phase of the grating pattern."""
        self.phase = 0
        self.curr_time = 0

        for i, p in enumerate(self.init_pos):
            physics.bind(self.cylinders[i]).mocap_pos = (
                p.real,
                p.imag,
                self.height / 2,
            )

    def step(self, dt, physics):
        """Steps the phase of the grating pattern and updates the position of the cylinders."""

        if self.curr_time % 1 < 1 / 2:
            self.phase -= dt * self.ang_speed
        else:
            self.phase += dt * self.ang_speed

        self.curr_time += dt

        pos = np.exp(self.phase * 1j) * self.init_pos

        for i, p in enumerate(pos):
            physics.bind(self.cylinders[i]).mocap_pos = (
                p.real,
                p.imag,
                self.height / 2,
            )


class LoomingTerrain(BaseArena):
    def __init__(
        self,
        move_direction = "towards",
        lateral_magnitude = 2,
        move_speed = 10,
        obj_radius = 4,
        init_ball_pos = (25, 4),
        size = (300, 300),
        friction = (1, 0.005, 0.0001),
    ):
        super().__init__()
        self.init_ball_pos = (*init_ball_pos, obj_radius)
        self.ball_pos = np.array(self.init_ball_pos, dtype="float32")
        self.friction = friction
        self.move_speed = move_speed
        self.curr_time = 0
        self.lateral_magnitude = lateral_magnitude
        self.move_direction = move_direction
        if move_direction == "towards":
            self.x_mult = -1
        elif move_direction == "away":
            self.x_mult = 1
        else:
            raise ValueError("Invalid move_direction")

        # Add ground
        ground_size = [*size, 1]
        chequered = self.root_element.asset.add(
            "texture",
            type="2d",
            builtin="checker",
            width=300,
            height=300,
            rgb1=(0.4, 0.4, 0.4),
            rgb2=(0.5, 0.5, 0.5),
        )
        grid = self.root_element.asset.add(
            "material",
            name="grid",
            texture=chequered,
            texrepeat=(60, 60),
            reflectance=0.1,
        )
        self.root_element.worldbody.add(
            "geom",
            type="plane",
            name="ground",
            material=grid,
            size=ground_size,
            friction=friction,
        )
        self.root_element.worldbody.add("body", name="b_plane")

        # Add ball
        obstacle = self.root_element.asset.add(
            "material", name="obstacle", reflectance=0.1
        )
        self.root_element.worldbody.add(
            "body", name="ball_mocap", mocap=True, pos=self.ball_pos, gravcomp=1
        )
        self.object_body = self.root_element.find("body", "ball_mocap")
        self.object_body.add(
            "geom",
            name="ball",
            type="sphere",
            size=(obj_radius, obj_radius),
            rgba=(0.0, 0.0, 0.0, 1),
            material=obstacle,
        )

        self.birdeye_cam = self.root_element.worldbody.add(
            "camera",
            name="birdeye_cam",
            mode="fixed",
            pos=(15, 0, 35),
            euler=(0, 0, 0),
            fovy=45,
        )
    
    def get_spawn_position(self, rel_pos, rel_angle):
        return rel_pos, rel_angle

    def step(self, dt, physics):
        heading_vec = np.array(
            [1.0 * self.lateral_magnitude * self.x_mult, 0]
        )
        #heading_vec /= np.linalg.norm(heading_vec)
        self.ball_pos[:2] += self.move_speed * heading_vec * dt
        physics.bind(self.object_body).mocap_pos = self.ball_pos
        self.curr_time += dt

    def reset(self, physics):
        self.curr_time = 0
        self.ball_pos = np.array(self.init_ball_pos, dtype="float32")
        physics.bind(self.object_body).mocap_pos = self.ball_pos
