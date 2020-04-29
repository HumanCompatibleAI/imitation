import numpy as np

from imitation.envs.examples.airl_envs.dynamic_mjc.model_builder import MJCModel


def block_push(object_pos=(0, 0, 0), goal_pos=(0, 0, 0)):
    mjcmodel = MJCModel("block_push")
    mjcmodel.root.compiler(inertiafromgeom="true", angle="radian", coordinate="local")
    mjcmodel.root.option(
        timestep="0.01", gravity="0 0 0", iterations="20", integrator="Euler"
    )
    default = mjcmodel.root.default()
    default.joint(armature="0.04", damping=1, limited="true")
    default.geom(
        friction=".8 .1 .1",
        density="300",
        margin="0.002",
        condim="1",
        contype="1",
        conaffinity="1",
    )

    worldbody = mjcmodel.root.worldbody()

    palm = worldbody.body(name="palm", pos=[0, 0, 0])
    palm.geom(
        name="palm_geom", type="capsule", fromto=[0, -0.1, 0, 0, 0.1, 0], size=0.12
    )
    proximal1 = palm.body(name="proximal_1", pos=[0, 0, 0])
    proximal1.joint(
        name="proximal_j_1",
        type="hinge",
        pos=[0, 0, 0],
        axis=[0, 1, 0],
        range=[-2.5, 2.3],
    )
    proximal1.geom(
        type="capsule", fromto=[0, 0, 0, 0.4, 0, 0], size=0.06, contype=1, conaffinity=1
    )
    distal1 = proximal1.body(name="distal_1", pos=[0.4, 0, 0])
    distal1.joint(
        name="distal_j_1",
        type="hinge",
        pos="0 0 0",
        axis="0 1 0",
        range="-2.3213 2.3",
        damping="1.0",
    )
    distal1.geom(
        type="capsule",
        fromto="0 0 0 0.4 0 0",
        size="0.06",
        contype="1",
        conaffinity="1",
    )
    distal2 = distal1.body(name="distal_2", pos=[0.4, 0, 0])
    distal2.joint(
        name="distal_j_2",
        type="hinge",
        pos="0 0 0",
        axis="0 1 0",
        range="-2.3213 2.3",
        damping="1.0",
    )
    distal2.geom(
        type="capsule",
        fromto="0 0 0 0.4 0 0",
        size="0.06",
        contype="1",
        conaffinity="1",
    )
    distal4 = distal2.body(name="distal_4", pos=[0.4, 0, 0])
    distal4.site(name="tip arml", pos="0.1 0 -0.2", size="0.01")
    distal4.site(name="tip armr", pos="0.1 0 0.2", size="0.01")
    distal4.joint(
        name="distal_j_3",
        type="hinge",
        pos="0 0 0",
        axis="1 0 0",
        range="-3.3213 3.3",
        damping="0.5",
    )
    distal4.geom(
        type="capsule",
        fromto="0 0 -0.2 0 0 0.2",
        size="0.04",
        contype="1",
        conaffinity="1",
    )
    distal4.geom(
        type="capsule",
        fromto="0 0 -0.2 0.2 0 -0.2",
        size="0.04",
        contype="1",
        conaffinity="1",
    )
    distal4.geom(
        type="capsule",
        fromto="0 0 0.2 0.2 0 0.2",
        size="0.04",
        contype="1",
        conaffinity="1",
    )

    obj = worldbody.body(name="obj", pos=object_pos)
    obj.geom(
        rgba="1. 1. 1. 1",
        type="box",
        size="0.05 0.05 0.05",
        density="0.00001",
        contype="1",
        conaffinity="1",
    )
    obj.joint(
        name="obj_slidez",
        type="slide",
        pos="0.025 0.025 0.025",
        axis="0 0 1",
        range="-10.3213 10.3",
        damping="0.5",
    )
    obj.joint(
        name="obj_slidex",
        type="slide",
        pos="0.025 0.025 0.025",
        axis="1 0 0",
        range="-10.3213 10.3",
        damping="0.5",
    )
    distal10 = obj.body(name="distal_10", pos=[0, 0, 0])
    distal10.site(name="obj_pos", pos=[0.025, 0.025, 0.025], size=0.01)

    goal = worldbody.body(name="goal", pos=goal_pos)
    goal.geom(
        rgba="1. 0. 0. 1",
        type="box",
        size="0.1 0.1 0.1",
        density="0.00001",
        contype="0",
        conaffinity="0",
    )
    distal11 = goal.body(name="distal_11", pos=[0, 0, 0])
    distal11.site(name="goal_pos", pos=[0.05, 0.05, 0.05], size=0.01)

    actuator = mjcmodel.root.actuator()
    actuator.motor(joint="proximal_j_1", ctrlrange="-2 2", ctrllimited="true")
    actuator.motor(joint="distal_j_1", ctrlrange="-2 2", ctrllimited="true")
    actuator.motor(joint="distal_j_2", ctrlrange="-2 2", ctrllimited="true")
    actuator.motor(joint="distal_j_3", ctrlrange="-2 2", ctrllimited="true")

    return mjcmodel


EAST = 0
WEST = 1
NORTH = 2
SOUTH = 3


def twod_corridor(direction=EAST, length=1.2):
    mjcmodel = MJCModel("twod_corridor")
    mjcmodel.root.compiler(inertiafromgeom="true", angle="radian", coordinate="local")
    mjcmodel.root.option(
        timestep="0.01", gravity="0 0 0", iterations="20", integrator="Euler"
    )
    default = mjcmodel.root.default()
    default.joint(damping=1, limited="false")
    default.geom(
        friction=".5 .1 .1",
        density="1000",
        margin="0.002",
        condim="1",
        contype="2",
        conaffinity="1",
    )

    worldbody = mjcmodel.root.worldbody()

    particle = worldbody.body(name="particle", pos=[0, 0, 0])
    particle.geom(
        name="particle_geom",
        type="sphere",
        size="0.03",
        rgba="0.0 0.0 1.0 1",
        contype=1,
    )
    particle.site(name="particle_site", pos=[0, 0, 0], size=0.01)
    particle.joint(name="ball_x", type="slide", pos=[0, 0, 0], axis=[1, 0, 0])
    particle.joint(name="ball_y", type="slide", pos=[0, 0, 0], axis=[0, 1, 0])

    pos = np.array([0.0, 0, 0])
    if direction == EAST or direction == WEST:
        pos[0] = length - 0.1
    else:
        pos[1] = length - 0.1
    if direction == WEST or direction == SOUTH:
        pos = -pos

    target = worldbody.body(name="target", pos=pos)
    target.geom(
        name="target_geom",
        conaffinity=2,
        type="sphere",
        size=0.02,
        rgba=[0, 0.9, 0.1, 1],
    )

    # arena
    if direction == EAST:
        L = -0.1
        R = length
        U = 0.1
        D = -0.1
    elif direction == WEST:
        L = -length
        R = 0.1
        U = 0.1
        D = -0.1
    elif direction == SOUTH:
        L = -0.1
        R = 0.1
        U = 0.1
        D = -length
    elif direction == NORTH:
        L = -0.1
        R = 0.1
        U = length
        D = -0.1

    worldbody.geom(
        conaffinity=1,
        fromto=[L, D, 0.01, R, D, 0.01],
        name="sideS",
        rgba="0.9 0.4 0.6 1",
        size=0.02,
        type="capsule",
    )
    worldbody.geom(
        conaffinity=1,
        fromto=[R, D, 0.01, R, U, 0.01],
        name="sideE",
        rgba="0.9 0.4 0.6 1",
        size=".02",
        type="capsule",
    )
    worldbody.geom(
        conaffinity=1,
        fromto=[L, U, 0.01, R, U, 0.01],
        name="sideN",
        rgba="0.9 0.4 0.6 1",
        size=".02",
        type="capsule",
    )
    worldbody.geom(
        conaffinity=1,
        fromto=[L, D, 0.01, L, U, 0.01],
        name="sideW",
        rgba="0.9 0.4 0.6 1",
        size=".02",
        type="capsule",
    )

    actuator = mjcmodel.root.actuator()
    actuator.motor(joint="ball_x", ctrlrange=[-1.0, 1.0], ctrllimited=True)
    actuator.motor(joint="ball_y", ctrlrange=[-1.0, 1.0], ctrllimited=True)

    return mjcmodel


LEFT = 0
RIGHT = 1


def point_mass_maze(direction=RIGHT, length=1.2, borders=True):
    mjcmodel = MJCModel("twod_maze")
    mjcmodel.root.compiler(inertiafromgeom="true", angle="radian", coordinate="local")
    mjcmodel.root.option(
        timestep="0.01", gravity="0 0 0", iterations="20", integrator="Euler"
    )
    default = mjcmodel.root.default()
    default.joint(damping=1, limited="false")
    default.geom(
        friction=".5 .1 .1",
        density="1000",
        margin="0.002",
        condim="1",
        contype="2",
        conaffinity="1",
    )

    worldbody = mjcmodel.root.worldbody()

    particle = worldbody.body(name="particle", pos=[length / 2, 0, 0])
    particle.geom(
        name="particle_geom",
        type="sphere",
        size="0.03",
        rgba="0.0 0.0 1.0 1",
        contype=1,
    )
    particle.site(name="particle_site", pos=[0, 0, 0], size=0.01)
    particle.joint(name="ball_x", type="slide", pos=[0, 0, 0], axis=[1, 0, 0])
    particle.joint(name="ball_y", type="slide", pos=[0, 0, 0], axis=[0, 1, 0])

    target = worldbody.body(name="target", pos=[length / 2, length - 0.1, 0])
    target.geom(
        name="target_geom",
        conaffinity=2,
        type="sphere",
        size=0.02,
        rgba=[0, 0.9, 0.1, 1],
    )

    L = -0.1
    R = length
    U = length
    D = -0.1

    if borders:
        worldbody.geom(
            conaffinity=1,
            fromto=[L, D, 0.01, R, D, 0.01],
            name="sideS",
            rgba="0.9 0.4 0.6 1",
            size=".02",
            type="capsule",
        )
        worldbody.geom(
            conaffinity=1,
            fromto=[R, D, 0.01, R, U, 0.01],
            name="sideE",
            rgba="0.9 0.4 0.6 1",
            size=".02",
            type="capsule",
        )
        worldbody.geom(
            conaffinity=1,
            fromto=[L, U, 0.01, R, U, 0.01],
            name="sideN",
            rgba="0.9 0.4 0.6 1",
            size=".02",
            type="capsule",
        )
        worldbody.geom(
            conaffinity=1,
            fromto=[L, D, 0.01, L, U, 0.01],
            name="sideW",
            rgba="0.9 0.4 0.6 1",
            size=".02",
            type="capsule",
        )

    # arena
    if direction == LEFT:
        BL = -0.1
        BR = length * 2 / 3
        BH = length / 2
    else:
        BL = length * 1 / 3
        BR = length
        BH = length / 2

    worldbody.geom(
        conaffinity=1,
        fromto=[BL, BH, 0.01, BR, BH, 0.01],
        name="barrier",
        rgba="0.9 0.4 0.6 1",
        size=".02",
        type="capsule",
    )

    actuator = mjcmodel.root.actuator()
    actuator.motor(joint="ball_x", ctrlrange=[-1.0, 1.0], ctrllimited=True)
    actuator.motor(joint="ball_y", ctrlrange=[-1.0, 1.0], ctrllimited=True)

    return mjcmodel


def ant_maze(direction=RIGHT, length=6.0):
    mjcmodel = MJCModel("ant_maze")
    mjcmodel.root.compiler(inertiafromgeom="true", angle="degree", coordinate="local")
    mjcmodel.root.option(
        timestep="0.01", gravity="0 0 -9.8", iterations="20", integrator="Euler"
    )

    assets = mjcmodel.root.asset()
    assets.texture(
        builtin="gradient",
        height="100",
        rgb1="1 1 1",
        rgb2="0 0 0",
        type="skybox",
        width="100",
    )
    assets.texture(
        builtin="flat",
        height="1278",
        mark="cross",
        markrgb="1 1 1",
        name="texgeom",
        random="0.01",
        rgb1="0.8 0.6 0.4",
        rgb2="0.8 0.6 0.4",
        type="cube",
        width="127",
    )
    assets.texture(
        builtin="checker",
        height="100",
        name="texplane",
        rgb1="0 0 0",
        rgb2="0.8 0.8 0.8",
        type="2d",
        width="100",
    )
    assets.material(
        name="MatPlane",
        reflectance="0.5",
        shininess="1",
        specular="1",
        texrepeat="60 60",
        texture="texplane",
    )
    assets.material(name="geom", texture="texgeom", texuniform="true")

    default = mjcmodel.root.default()
    default.joint(armature="1", damping=1, limited="true")
    default.geom(
        friction="1 0.5 0.5", density="5.0", margin="0.01", condim="3", conaffinity="0"
    )

    worldbody = mjcmodel.root.worldbody()

    ant = worldbody.body(name="ant", pos=[length / 2, 1.0, 0.05])
    ant.geom(name="torso_geom", pos=[0, 0, 0], size="0.25", type="sphere")
    ant.joint(
        armature="0",
        damping="0",
        limited="false",
        margin="0.01",
        name="root",
        pos=[0, 0, 0],
        type="free",
    )

    front_left_leg = ant.body(name="front_left_leg", pos=[0, 0, 0])
    front_left_leg.geom(
        fromto=[0.0, 0.0, 0.0, 0.2, 0.2, 0.0],
        name="aux_1_geom",
        size="0.08",
        type="capsule",
    )
    aux_1 = front_left_leg.body(name="aux_1", pos=[0.2, 0.2, 0])
    aux_1.joint(
        axis=[0, 0, 1], name="hip_1", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge"
    )
    aux_1.geom(
        fromto=[0.0, 0.0, 0.0, 0.2, 0.2, 0.0],
        name="left_leg_geom",
        size="0.08",
        type="capsule",
    )
    ankle_1 = aux_1.body(pos=[0.2, 0.2, 0])
    ankle_1.joint(
        axis=[-1, 1, 0],
        name="ankle_1",
        pos=[0.0, 0.0, 0.0],
        range=[30, 70],
        type="hinge",
    )
    ankle_1.geom(
        fromto=[0.0, 0.0, 0.0, 0.4, 0.4, 0.0],
        name="left_ankle_geom",
        size="0.08",
        type="capsule",
    )

    front_right_leg = ant.body(name="front_right_leg", pos=[0, 0, 0])
    front_right_leg.geom(
        fromto=[0.0, 0.0, 0.0, -0.2, 0.2, 0.0],
        name="aux_2_geom",
        size="0.08",
        type="capsule",
    )
    aux_2 = front_right_leg.body(name="aux_2", pos=[-0.2, 0.2, 0])
    aux_2.joint(
        axis=[0, 0, 1], name="hip_2", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge"
    )
    aux_2.geom(
        fromto=[0.0, 0.0, 0.0, -0.2, 0.2, 0.0],
        name="right_leg_geom",
        size="0.08",
        type="capsule",
    )
    ankle_2 = aux_2.body(pos=[-0.2, 0.2, 0])
    ankle_2.joint(
        axis=[1, 1, 0],
        name="ankle_2",
        pos=[0.0, 0.0, 0.0],
        range=[-70, -30],
        type="hinge",
    )
    ankle_2.geom(
        fromto=[0.0, 0.0, 0.0, -0.4, 0.4, 0.0],
        name="right_ankle_geom",
        size="0.08",
        type="capsule",
    )

    back_left_leg = ant.body(name="back_left_leg", pos=[0, 0, 0])
    back_left_leg.geom(
        fromto=[0.0, 0.0, 0.0, -0.2, -0.2, 0.0],
        name="aux_3_geom",
        size="0.08",
        type="capsule",
    )
    aux_3 = back_left_leg.body(name="aux_3", pos=[-0.2, -0.2, 0])
    aux_3.joint(
        axis=[0, 0, 1], name="hip_3", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge"
    )
    aux_3.geom(
        fromto=[0.0, 0.0, 0.0, -0.2, -0.2, 0.0],
        name="backleft_leg_geom",
        size="0.08",
        type="capsule",
    )
    ankle_3 = aux_3.body(pos=[-0.2, -0.2, 0])
    ankle_3.joint(
        axis=[-1, 1, 0],
        name="ankle_3",
        pos=[0.0, 0.0, 0.0],
        range=[-70, -30],
        type="hinge",
    )
    ankle_3.geom(
        fromto=[0.0, 0.0, 0.0, -0.4, -0.4, 0.0],
        name="backleft_ankle_geom",
        size="0.08",
        type="capsule",
    )

    back_right_leg = ant.body(name="back_right_leg", pos=[0, 0, 0])
    back_right_leg.geom(
        fromto=[0.0, 0.0, 0.0, 0.2, -0.2, 0.0],
        name="aux_4_geom",
        size="0.08",
        type="capsule",
    )
    aux_4 = back_right_leg.body(name="aux_4", pos=[0.2, -0.2, 0])
    aux_4.joint(
        axis=[0, 0, 1], name="hip_4", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge"
    )
    aux_4.geom(
        fromto=[0.0, 0.0, 0.0, 0.2, -0.2, 0.0],
        name="backright_leg_geom",
        size="0.08",
        type="capsule",
    )
    ankle_4 = aux_4.body(pos=[0.2, -0.2, 0])
    ankle_4.joint(
        axis=[1, 1, 0],
        name="ankle_4",
        pos=[0.0, 0.0, 0.0],
        range=[30, 70],
        type="hinge",
    )
    ankle_4.geom(
        fromto=[0.0, 0.0, 0.0, 0.4, -0.4, 0.0],
        name="backright_ankle_geom",
        size="0.08",
        type="capsule",
    )

    target = worldbody.body(name="target", pos=[length / 2, length - 0.2, -0.5])
    target.geom(
        name="target_geom",
        conaffinity=2,
        type="sphere",
        size=0.2,
        rgba=[0, 0.9, 0.1, 1],
    )

    L = length / 2
    h = 0.75
    w = 0.05

    worldbody.geom(
        conaffinity=1,
        name="sideS",
        rgba="0.9 0.4 0.6 1",
        size=[L, w, h],
        pos=[length / 2, 0, 0],
        type="box",
    )
    worldbody.geom(
        conaffinity=1,
        name="sideE",
        rgba="0.9 0.4 0.6 1",
        size=[w, L, h],
        pos=[length, length / 2, 0],
        type="box",
    )
    worldbody.geom(
        conaffinity=1,
        name="sideN",
        rgba="0.9 0.4 0.6 1",
        size=[L, w, h],
        pos=[length / 2, length, 0],
        type="box",
    )
    worldbody.geom(
        conaffinity=1,
        name="sideW",
        rgba="0.9 0.4 0.6 1",
        size=[w, L, h],
        pos=[0, length / 2, 0],
        type="box",
    )

    # arena
    if direction == LEFT:
        bx, by, bz = (length / 3, length / 2, 0)
    else:
        bx, by, bz = (length * 2 / 3, length / 2, 0)

    worldbody.geom(
        conaffinity=1,
        name="barrier",
        rgba="0.9 0.4 0.6 1",
        size=[L * 2 / 3, w, h],
        pos=[bx, by, bz],
        type="box",
    )
    worldbody.geom(
        conaffinity="1",
        condim="3",
        material="MatPlane",
        name="floor",
        pos=[length / 2, length / 2, -h + w],
        rgba="0.8 0.9 0.8 1",
        size="40 40 40",
        type="plane",
    )

    actuator = mjcmodel.root.actuator()
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_4", gear="50")
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_4", gear="50")
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_1", gear="50")
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_1", gear="50")
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_2", gear="50")
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_2", gear="50")
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_3", gear="50")
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_3", gear="50")

    return mjcmodel


def ant_maze_corridor(direction=RIGHT, height=6.0, width=10.0):
    mjcmodel = MJCModel("ant_maze_corridor")
    mjcmodel.root.compiler(inertiafromgeom="true", angle="degree", coordinate="local")
    mjcmodel.root.option(
        timestep="0.01", gravity="0 0 -9.8", iterations="20", integrator="Euler"
    )

    assets = mjcmodel.root.asset()
    assets.texture(
        builtin="gradient",
        height="100",
        rgb1="1 1 1",
        rgb2="0 0 0",
        type="skybox",
        width="100",
    )
    assets.texture(
        builtin="flat",
        height="1278",
        mark="cross",
        markrgb="1 1 1",
        name="texgeom",
        random="0.01",
        rgb1="0.8 0.6 0.4",
        rgb2="0.8 0.6 0.4",
        type="cube",
        width="127",
    )
    assets.texture(
        builtin="checker",
        height="100",
        name="texplane",
        rgb1="0 0 0",
        rgb2="0.8 0.8 0.8",
        type="2d",
        width="100",
    )
    assets.material(
        name="MatPlane",
        reflectance="0.5",
        shininess="1",
        specular="1",
        texrepeat="60 60",
        texture="texplane",
    )
    assets.material(name="geom", texture="texgeom", texuniform="true")

    default = mjcmodel.root.default()
    default.joint(armature="1", damping=1, limited="true")
    default.geom(
        friction="1 0.5 0.5", density="5.0", margin="0.01", condim="3", conaffinity="0"
    )

    worldbody = mjcmodel.root.worldbody()

    ant = worldbody.body(name="ant", pos=[height / 2, 1.0, 0.05])
    ant.geom(name="torso_geom", pos=[0, 0, 0], size="0.25", type="sphere")
    ant.joint(
        armature="0",
        damping="0",
        limited="false",
        margin="0.01",
        name="root",
        pos=[0, 0, 0],
        type="free",
    )

    front_left_leg = ant.body(name="front_left_leg", pos=[0, 0, 0])
    front_left_leg.geom(
        fromto=[0.0, 0.0, 0.0, 0.2, 0.2, 0.0],
        name="aux_1_geom",
        size="0.08",
        type="capsule",
    )
    aux_1 = front_left_leg.body(name="aux_1", pos=[0.2, 0.2, 0])
    aux_1.joint(
        axis=[0, 0, 1], name="hip_1", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge"
    )
    aux_1.geom(
        fromto=[0.0, 0.0, 0.0, 0.2, 0.2, 0.0],
        name="left_leg_geom",
        size="0.08",
        type="capsule",
    )
    ankle_1 = aux_1.body(pos=[0.2, 0.2, 0])
    ankle_1.joint(
        axis=[-1, 1, 0],
        name="ankle_1",
        pos=[0.0, 0.0, 0.0],
        range=[30, 70],
        type="hinge",
    )
    ankle_1.geom(
        fromto=[0.0, 0.0, 0.0, 0.4, 0.4, 0.0],
        name="left_ankle_geom",
        size="0.08",
        type="capsule",
    )

    front_right_leg = ant.body(name="front_right_leg", pos=[0, 0, 0])
    front_right_leg.geom(
        fromto=[0.0, 0.0, 0.0, -0.2, 0.2, 0.0],
        name="aux_2_geom",
        size="0.08",
        type="capsule",
    )
    aux_2 = front_right_leg.body(name="aux_2", pos=[-0.2, 0.2, 0])
    aux_2.joint(
        axis=[0, 0, 1], name="hip_2", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge"
    )
    aux_2.geom(
        fromto=[0.0, 0.0, 0.0, -0.2, 0.2, 0.0],
        name="right_leg_geom",
        size="0.08",
        type="capsule",
    )
    ankle_2 = aux_2.body(pos=[-0.2, 0.2, 0])
    ankle_2.joint(
        axis=[1, 1, 0],
        name="ankle_2",
        pos=[0.0, 0.0, 0.0],
        range=[-70, -30],
        type="hinge",
    )
    ankle_2.geom(
        fromto=[0.0, 0.0, 0.0, -0.4, 0.4, 0.0],
        name="right_ankle_geom",
        size="0.08",
        type="capsule",
    )

    back_left_leg = ant.body(name="back_left_leg", pos=[0, 0, 0])
    back_left_leg.geom(
        fromto=[0.0, 0.0, 0.0, -0.2, -0.2, 0.0],
        name="aux_3_geom",
        size="0.08",
        type="capsule",
    )
    aux_3 = back_left_leg.body(name="aux_3", pos=[-0.2, -0.2, 0])
    aux_3.joint(
        axis=[0, 0, 1], name="hip_3", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge"
    )
    aux_3.geom(
        fromto=[0.0, 0.0, 0.0, -0.2, -0.2, 0.0],
        name="backleft_leg_geom",
        size="0.08",
        type="capsule",
    )
    ankle_3 = aux_3.body(pos=[-0.2, -0.2, 0])
    ankle_3.joint(
        axis=[-1, 1, 0],
        name="ankle_3",
        pos=[0.0, 0.0, 0.0],
        range=[-70, -30],
        type="hinge",
    )
    ankle_3.geom(
        fromto=[0.0, 0.0, 0.0, -0.4, -0.4, 0.0],
        name="backleft_ankle_geom",
        size="0.08",
        type="capsule",
    )

    back_right_leg = ant.body(name="back_right_leg", pos=[0, 0, 0])
    back_right_leg.geom(
        fromto=[0.0, 0.0, 0.0, 0.2, -0.2, 0.0],
        name="aux_4_geom",
        size="0.08",
        type="capsule",
    )
    aux_4 = back_right_leg.body(name="aux_4", pos=[0.2, -0.2, 0])
    aux_4.joint(
        axis=[0, 0, 1], name="hip_4", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge"
    )
    aux_4.geom(
        fromto=[0.0, 0.0, 0.0, 0.2, -0.2, 0.0],
        name="backright_leg_geom",
        size="0.08",
        type="capsule",
    )
    ankle_4 = aux_4.body(pos=[0.2, -0.2, 0])
    ankle_4.joint(
        axis=[1, 1, 0],
        name="ankle_4",
        pos=[0.0, 0.0, 0.0],
        range=[30, 70],
        type="hinge",
    )
    ankle_4.geom(
        fromto=[0.0, 0.0, 0.0, 0.4, -0.4, 0.0],
        name="backright_ankle_geom",
        size="0.08",
        type="capsule",
    )

    target = worldbody.body(name="target", pos=[height / 2, width - 1.0, -0.5])
    target.geom(
        name="target_geom",
        conaffinity=2,
        type="sphere",
        size=0.2,
        rgba=[0, 0.9, 0.1, 1],
    )

    L = height / 2
    h = 0.75
    w = 0.05

    worldbody.geom(
        conaffinity=1,
        name="sideS",
        rgba="0.9 0.4 0.6 1",
        size=[L, w, h],
        pos=[height / 2, 0, 0],
        type="box",
    )
    worldbody.geom(
        conaffinity=1,
        name="sideE",
        rgba="0.9 0.4 0.6 1",
        size=[w, width / 2, h],
        pos=[height, width / 2, 0],
        type="box",
    )
    worldbody.geom(
        conaffinity=1,
        name="sideN",
        rgba="0.9 0.4 0.6 1",
        size=[L, w, h],
        pos=[height / 2, width, 0],
        type="box",
    )
    worldbody.geom(
        conaffinity=1,
        name="sideW",
        rgba="0.9 0.4 0.6 1",
        size=[w, width / 2, h],
        pos=[0, width / 2, 0],
        type="box",
    )

    # arena
    wall_ratio = 0.55  # 2.0/3
    if direction == LEFT:
        bx, by, bz = (height * (wall_ratio / 2), width / 2, 0)
    else:
        bx, by, bz = (height * (1 - wall_ratio / 2), width / 2, 0)

    worldbody.geom(
        conaffinity=1,
        name="barrier",
        rgba="0.9 0.4 0.6 1",
        size=[L * (wall_ratio), w, h],
        pos=[bx, by, bz],
        type="box",
    )
    worldbody.geom(
        conaffinity="1",
        condim="3",
        material="MatPlane",
        name="floor",
        pos=[height / 2, height / 2, -h + w],
        rgba="0.8 0.9 0.8 1",
        size="40 40 40",
        type="plane",
    )

    actuator = mjcmodel.root.actuator()
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_4", gear="30")
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_4", gear="30")
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_1", gear="30")
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_1", gear="30")
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_2", gear="30")
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_2", gear="30")
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_3", gear="30")
    actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_3", gear="30")

    return mjcmodel


def pusher(goal_pos=np.array([0.45, -0.05, -0.323])):
    mjcmodel = MJCModel("pusher")
    mjcmodel.root.compiler(inertiafromgeom="true", angle="radian", coordinate="local")
    mjcmodel.root.option(
        timestep="0.01", gravity="0 0 0", iterations="20", integrator="Euler"
    )
    default = mjcmodel.root.default()
    default.joint(armature=0.04, damping=1, limited=False)
    default.geom(
        friction=[0.8, 0.1, 0.1],
        density=300,
        margin=0.002,
        condim=1,
        contype=0,
        conaffinity=0,
    )

    worldbody = mjcmodel.root.worldbody()
    worldbody.light(diffuse=[0.5, 0.5, 0.5], pos=[0, 0, 3], dir=[0, 0, -1])
    worldbody.geom(
        name="table",
        type="plane",
        pos=[0, 0.5, -0.325],
        size=[1, 1, 0.1],
        contype=1,
        conaffinity=1,
    )

    r_shoulder_pan_link = worldbody.body(name="r_shoulder_pan_link", pos=[0, -0.6, 0])
    r_shoulder_pan_link.geom(
        name="e1",
        type="sphere",
        rgba=[0.6, 0.6, 0.6, 1],
        pos=[-0.06, 0.05, 0.2],
        size=0.05,
    )
    r_shoulder_pan_link.geom(
        name="e2",
        type="sphere",
        rgba=[0.6, 0.6, 0.6, 1],
        pos=[0.06, 0.05, 0.2],
        size=0.05,
    )
    r_shoulder_pan_link.geom(
        name="e1p",
        type="sphere",
        rgba=[0.1, 0.1, 0.1, 1],
        pos=[-0.06, 0.09, 0.2],
        size=0.03,
    )
    r_shoulder_pan_link.geom(
        name="e2p",
        type="sphere",
        rgba=[0.1, 0.1, 0.1, 1],
        pos=[0.06, 0.09, 0.2],
        size=0.03,
    )
    r_shoulder_pan_link.geom(
        name="sp", type="capsule", fromto=[0, 0, -0.4, 0, 0, 0.2], size=0.1
    )

    r_shoulder_pan_link.joint(
        name="r_shoulder_pan_joint",
        type="hinge",
        pos=[0, 0, 0],
        axis=[0, 0, 1],
        range=[-2.2854, 1.714602],
        damping=1.0,
    )

    r_shoulder_lift_link = r_shoulder_pan_link.body(
        name="r_shoulder_lift_link", pos=[0.1, 0, 0]
    )
    r_shoulder_lift_link.geom(
        name="s1", type="capsule", fromto="0 -0.1 0 0 0.1 0", size="0.1"
    )
    r_shoulder_lift_link.joint(
        name="r_shoulder_lift_joint",
        type="hinge",
        pos="0 0 0",
        axis="0 1 0",
        range="-0.5236 1.3963",
        damping="1.0",
    )

    r_upper_arm_roll_link = r_shoulder_lift_link.body(
        name="r_upper_arm_roll_link", pos=[0, 0, 0]
    )
    r_upper_arm_roll_link.geom(
        name="uar", type="capsule", fromto="-0.1 0 0 0.1 0 0", size="0.02"
    )
    r_upper_arm_roll_link.joint(
        name="r_upper_arm_roll_joint",
        type="hinge",
        pos="0 0 0",
        axis="1 0 0",
        range="-1.5 1.7",
        damping="0.1",
    )

    r_upper_arm_link = r_upper_arm_roll_link.body(
        name="r_upper_arm_link", pos=[0, 0, 0]
    )
    r_upper_arm_link.geom(
        name="ua", type="capsule", fromto="0 0 0 0.4 0 0", size="0.06"
    )

    r_elbow_flex_link = r_upper_arm_link.body(name="r_elbow_flex_link", pos=[0.4, 0, 0])
    r_elbow_flex_link.geom(
        name="ef", type="capsule", fromto="0 -0.02 0 0.0 0.02 0", size="0.06"
    )
    r_elbow_flex_link.joint(
        name="r_elbow_flex_joint",
        type="hinge",
        pos="0 0 0",
        axis="0 1 0",
        range="-2.3213 0",
        damping="0.1",
    )

    r_forearm_roll_link = r_elbow_flex_link.body(
        name="r_forearm_roll_link", pos=[0, 0, 0]
    )
    r_forearm_roll_link.geom(
        name="fr", type="capsule", fromto="-0.1 0 0 0.1 0 0", size="0.02"
    )
    r_forearm_roll_link.joint(
        name="r_forearm_roll_joint",
        type="hinge",
        limited="true",
        pos="0 0 0",
        axis="1 0 0",
        damping=".1",
        range="-1.5 1.5",
    )

    r_forearm_link = r_forearm_roll_link.body(name="r_forearm_link", pos=[0, 0, 0])
    r_forearm_link.geom(
        name="fa", type="capsule", fromto="0 0 0 0.291 0 0", size="0.05"
    )

    r_wrist_flex_link = r_forearm_link.body(name="r_wrist_flex_link", pos=[0.321, 0, 0])
    r_wrist_flex_link.geom(
        name="wf", type="capsule", fromto="0 -0.02 0 0 0.02 0", size="0.01"
    )
    r_wrist_flex_link.joint(
        name="r_wrist_flex_joint",
        type="hinge",
        pos="0 0 0",
        axis="0 1 0",
        range="-1.094 0",
        damping=".1",
    )

    r_wrist_roll_link = r_wrist_flex_link.body(name="r_wrist_roll_link", pos=[0, 0, 0])
    r_wrist_roll_link.joint(
        name="r_wrist_roll_joint",
        type="hinge",
        pos="0 0 0",
        limited="true",
        axis="1 0 0",
        damping="0.1",
        range="-1.5 1.5",
    )
    r_wrist_roll_link.geom(
        type="capsule",
        fromto="0 -0.1 0. 0.0 +0.1 0",
        size="0.02",
        contype="1",
        conaffinity="1",
    )
    r_wrist_roll_link.geom(
        type="capsule",
        fromto="0 -0.1 0. 0.1 -0.1 0",
        size="0.02",
        contype="1",
        conaffinity="1",
    )
    r_wrist_roll_link.geom(
        type="capsule",
        fromto="0 +0.1 0. 0.1 +0.1 0",
        size="0.02",
        contype="1",
        conaffinity="1",
    )

    tips_arm = r_wrist_roll_link.body(name="tips_arm", pos=[0, 0, 0])
    tips_arm.geom(name="tip_arml", type="sphere", pos="0.1 -0.1 0.", size="0.01")
    tips_arm.geom(name="tip_armr", type="sphere", pos="0.1 0.1 0.", size="0.01")

    object_ = worldbody.body(name="object", pos=[0.0, 0.0, -0.275])
    object_.geom(
        rgba="1 1 1 1",
        type="cylinder",
        size="0.05 0.05 0.05",
        density="0.00001",
        conaffinity="0",
        contype=1,
    )
    object_.joint(
        name="obj_slidey",
        type="slide",
        pos="0 0 0",
        axis="0 1 0",
        range="-10.3213 10.3",
        damping="0.5",
    )
    object_.joint(
        name="obj_slidex",
        type="slide",
        pos="0 0 0",
        axis="1 0 0",
        range="-10.3213 10.3",
        damping="0.5",
    )

    goal = worldbody.body(name="goal", pos=goal_pos)
    goal.geom(
        rgba="1 0 0 1",
        type="cylinder",
        size="0.08 0.001 0.1",
        density="0.00001",
        contype="0",
        conaffinity="0",
    )
    goal.joint(
        name="goal_slidey",
        type="slide",
        pos="0 0 0",
        axis="0 1 0",
        range="-10.3213 10.3",
        damping="0.5",
    )
    goal.joint(
        name="goal_slidex",
        type="slide",
        pos="0 0 0",
        axis="1 0 0",
        range="-10.3213 10.3",
        damping="0.5",
    )

    actuator = mjcmodel.root.actuator()
    actuator.motor(
        joint="r_shoulder_pan_joint", ctrlrange=[-2.0, 2.0], ctrllimited=True
    )
    actuator.motor(
        joint="r_shoulder_lift_joint", ctrlrange=[-2.0, 2.0], ctrllimited=True
    )
    actuator.motor(
        joint="r_upper_arm_roll_joint", ctrlrange=[-2.0, 2.0], ctrllimited=True
    )
    actuator.motor(joint="r_elbow_flex_joint", ctrlrange=[-2.0, 2.0], ctrllimited=True)
    actuator.motor(
        joint="r_forearm_roll_joint", ctrlrange=[-2.0, 2.0], ctrllimited=True
    )
    actuator.motor(joint="r_wrist_flex_joint", ctrlrange=[-2.0, 2.0], ctrllimited=True)
    actuator.motor(joint="r_wrist_roll_joint", ctrlrange=[-2.0, 2.0], ctrllimited=True)

    return mjcmodel


def swimmer():
    mjcmodel = MJCModel("swimmer")
    mjcmodel.root.compiler(inertiafromgeom="true", angle="degree", coordinate="local")
    mjcmodel.root.option(
        timestep=0.01,
        viscosity=0.1,
        density=4000,
        integrator="RK4",
        collision="predefined",
    )
    default = mjcmodel.root.default()
    default.joint(armature=0.1)
    default.geom(
        rgba=[0.8, 0.6, 0.1, 1], condim=1, contype=1, conaffinity=1, material="geom"
    )

    asset = mjcmodel.root.asset()
    asset.texture(
        builtin="gradient",
        height=100,
        rgb1=[1, 1, 1],
        rgb2=[0, 0, 0],
        type="skybox",
        width=100,
    )
    asset.texture(
        builtin="flat",
        height=1278,
        mark="cross",
        markrgb=[1, 1, 1],
        name="texgeom",
        random=0.01,
        rgb1=[0.8, 0.6, 0.4],
        rgb2=[0.8, 0.6, 0.4],
        type="cube",
        width=127,
    )
    asset.texture(
        builtin="checker",
        height=100,
        name="texplane",
        rgb1=[0, 0, 0],
        rgb2=[0.8, 0.8, 0.8],
        type="2d",
        width=100,
    )
    asset.material(
        name="MatPlane",
        reflectance=0.5,
        shininess=1,
        specular=1,
        texrepeat=[30, 30],
        texture="texplane",
    )
    asset.material(name="geom", texture="texgeom", texuniform=True)

    worldbody = mjcmodel.root.worldbody()
    worldbody.light(
        cutoff=100,
        diffuse=[1, 1, 1],
        dir=[0, 0, -1.3],
        directional=True,
        exponent=1,
        pos=[0, 0, 1.3],
        specular=[0.1, 0.1, 0.1],
    )
    worldbody.geom(
        conaffinity=1,
        condim=3,
        material="MatPlane",
        name="floor",
        pos=[0, 0, -0.1],
        rgba=[0.8, 0.9, 0.9, 1],
        size=[40, 40, 0.1],
        type="plane",
    )
    torso = worldbody.body(name="torso", pos=[0, 0, 0])
    torso.geom(density=1000, fromto=[1.5, 0, 0, 0.5, 0, 0], size=0.1, type="capsule")
    torso.joint(axis=[1, 0, 0], name="slider1", pos=[0, 0, 0], type="slide")
    torso.joint(axis=[0, 1, 0], name="slider2", pos=[0, 0, 0], type="slide")
    torso.joint(axis=[0, 0, 1], name="rot", pos=[0, 0, 0], type="hinge")
    mid = torso.body(name="mid", pos=[0.5, 0, 0])
    mid.geom(density=1000, fromto=[0, 0, 0, -1, 0, 0], size=0.1, type="capsule")
    mid.joint(
        axis=[0, 0, 1],
        limited=True,
        name="rot2",
        pos=[0, 0, 0],
        range=[-100, 100],
        type="hinge",
    )
    back = mid.body(name="back", pos=[-1, 0, 0])
    back.geom(density=1000, fromto=[0, 0, 0, -1, 0, 0], size=0.1, type="capsule")
    back.joint(
        axis=[0, 0, 1],
        limited=True,
        name="rot3",
        pos=[0, 0, 0],
        range=[-100, 100],
        type="hinge",
    )

    actuator = mjcmodel.root.actuator()
    actuator.motor(ctrllimited=True, ctrlrange=[-1, 1], gear=150, joint="rot2")
    actuator.motor(ctrllimited=True, ctrlrange=[-1, 1], gear=150, joint="rot3")
    return mjcmodel


def swimmer_rllab():
    mjcmodel = MJCModel("swimmer")
    mjcmodel.root.compiler(inertiafromgeom="true", angle="degree", coordinate="local")
    mjcmodel.root.option(
        timestep=0.01,
        viscosity=0.1,
        density=4000,
        integrator="Euler",
        iterations=1000,
        collision="predefined",
    )

    custom = mjcmodel.root.custom()
    custom.numeric(name="frame_skip", data=50)

    default = mjcmodel.root.default()
    # default.joint(armature=0.1)
    default.geom(
        rgba=[0.8, 0.6, 0.1, 1], condim=1, contype=1, conaffinity=1, material="geom"
    )

    asset = mjcmodel.root.asset()
    asset.texture(
        builtin="gradient",
        height=100,
        rgb1=[1, 1, 1],
        rgb2=[0, 0, 0],
        type="skybox",
        width=100,
    )
    asset.texture(
        builtin="flat",
        height=1278,
        mark="cross",
        markrgb=[1, 1, 1],
        name="texgeom",
        random=0.01,
        rgb1=[0.8, 0.6, 0.4],
        rgb2=[0.8, 0.6, 0.4],
        type="cube",
        width=127,
    )
    asset.texture(
        builtin="checker",
        height=100,
        name="texplane",
        rgb1=[0, 0, 0],
        rgb2=[0.8, 0.8, 0.8],
        type="2d",
        width=100,
    )
    asset.material(
        name="MatPlane",
        reflectance=0.5,
        shininess=1,
        specular=1,
        texrepeat=[30, 30],
        texture="texplane",
    )
    asset.material(name="geom", texture="texgeom", texuniform=True)

    worldbody = mjcmodel.root.worldbody()
    worldbody.light(
        cutoff=100,
        diffuse=[1, 1, 1],
        dir=[0, 0, -1.3],
        directional=True,
        exponent=1,
        pos=[0, 0, 1.3],
        specular=[0.1, 0.1, 0.1],
    )
    worldbody.geom(
        conaffinity=1,
        condim=3,
        material="MatPlane",
        name="floor",
        pos=[0, 0, -0.1],
        rgba=[0.8, 0.9, 0.9, 1],
        size=[40, 40, 0.1],
        type="plane",
    )
    torso = worldbody.body(name="torso", pos=[0, 0, 0])
    torso.geom(density=1000, fromto=[1.5, 0, 0, 0.5, 0, 0], size=0.1, type="capsule")
    torso.joint(axis=[1, 0, 0], name="slider1", pos=[0, 0, 0], type="slide")
    torso.joint(axis=[0, 1, 0], name="slider2", pos=[0, 0, 0], type="slide")
    torso.joint(axis=[0, 0, 1], name="rot", pos=[0, 0, 0], type="hinge")
    mid = torso.body(name="mid", pos=[0.5, 0, 0])
    mid.geom(density=1000, fromto=[0, 0, 0, -1, 0, 0], size=0.1, type="capsule")
    mid.joint(
        axis=[0, 0, 1],
        limited=True,
        name="rot2",
        pos=[0, 0, 0],
        range=[-100, 100],
        type="hinge",
    )
    back = mid.body(name="back", pos=[-1, 0, 0])
    back.geom(density=1000, fromto=[0, 0, 0, -1, 0, 0], size=0.1, type="capsule")
    back.joint(
        axis=[0, 0, 1],
        limited=True,
        name="rot3",
        pos=[0, 0, 0],
        range=[-100, 100],
        type="hinge",
    )

    actuator = mjcmodel.root.actuator()
    actuator.motor(ctrllimited=True, ctrlrange=[-50, 50], joint="rot2")
    actuator.motor(ctrllimited=True, ctrlrange=[-50, 50], joint="rot3")
    return mjcmodel
