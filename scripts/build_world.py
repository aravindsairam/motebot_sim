"""Build the vacuum robot world and save it as a USD file.

Creates the world chosen with --world (house_layout.py hands out its layout), references the iRobot Create 3,
adds its sensors (2D lidar, RGB-D camera, IMU, bumper contact, cliff rays), and adds the
ROS 2 action graphs (clock, wheel drive, joint states, camera). Odometry, IMU, lidar, bumper
and cliff data are published by ros_publishers.py when the world is run through run_sim.py.

Usage (from a shell WITHOUT ROS sourced):
    cd ~/isaacsim && ./python.sh <repo>/sim/scripts/build_world.py                # the house
    cd ~/isaacsim && ./python.sh <repo>/sim/scripts/build_world.py --world home   # the home

Output: <repo>/sim/assets/vacuum_world.usd for the house, <world>_world.usd for any other world
(open it in Isaac Sim, or run run_sim.py --world <world>)
"""

import argparse
import math
import os
import sys

from isaacsim import SimulationApp

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.normpath(os.path.join(HERE, "..", "assets"))


def world_usd(world):
    """Where a world's USD file lives (the house keeps the name it has always had)."""
    return os.path.join(ASSETS, "vacuum_world.usd" if world == "house" else f"{world}_world.usd")


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--world", default="house", help="which world to build: house (default) or home")
parser.add_argument("--out", default=None, help="output USD path (default: the world's file in sim/assets)")
args, _ = parser.parse_known_args()
args.out = args.out or world_usd(args.world)
os.environ["VACUUM_WORLD"] = args.world  # read by house_layout.py when it is imported below

simulation_app = SimulationApp({"headless": True})

import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
import omni.graph.core as og  # noqa: E402
import omni.usd  # noqa: E402
import usdrt.Sdf  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdSemantics, UsdShade  # noqa: E402

sys.path.insert(0, HERE)
import house_layout as hl  # noqa: E402

ROBOT_PATH = "/World/create3"
# The chassis rigid body. Odometry must track this body, not the articulation root prim:
# with the root prim the odometry node follows whichever link PhysX picks as root.
BASE_LINK_PATH = ROBOT_PATH + "/base_link"
CASTER_COLLIDER_PATH = BASE_LINK_PATH + "/collisions/mesh_2"
WHEEL_CENTER_Z = 0.0402  # wheel axle height in the asset when base_link is at z = 0
CHASSIS_CENTER_OF_MASS = (0.025, 0.0, 0.03)  # m in base_link: 2.5 cm ahead of the axle, low (our choice)
LIDAR_PATH = BASE_LINK_PATH + "/laser"
CAMERA_LINK_PATH = BASE_LINK_PATH + "/camera_link"
CAMERA_PATH = CAMERA_LINK_PATH + "/front_camera"
IMU_LINK_PATH = BASE_LINK_PATH + "/imu_link"
IMU_PATH = IMU_LINK_PATH + "/imu_sensor"
BUMPER_PATH = BASE_LINK_PATH + "/bumper_contact"
CLIFF_PATH = BASE_LINK_PATH + "/cliff_rays"
SIM_RATE_HZ = 60.0  # run_sim.py default; the camera frame skip is derived from it
ROBOT_ASSET = "/Isaac/Robots/iRobot/Create3/create_3.usd"
LEFT_WHEEL_JOINT = "left_wheel_joint"
RIGHT_WHEEL_JOINT = "right_wheel_joint"


def make_materials(stage):
    """Create one UsdPreviewSurface material per entry in house_layout.MATERIALS."""
    mats = {}
    UsdGeom.Scope.Define(stage, "/World/Looks")
    for name, (rgb, roughness) in hl.MATERIALS.items():
        mat = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
        shader = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        if name in ("window_glass", "glass"):  # see-through
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.15)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        mats[name] = mat
    return mats


def add_dock_tag(stage):
    """Put the AprilTag picture on the front of the dock tower, as a flat square facing the room."""
    half = hl.DOCK_TAG_SIZE * 10.0 / 8.0 / 2.0  # the picture includes the white border: 10 cells for 8
    cx, cy = hl.to_world(hl.DOCK_TAG_HOUSE[0] + 0.001, hl.DOCK_TAG_HOUSE[1])  # 1 mm proud of the tower
    cz = hl.DOCK_TAG_HOUSE[2]
    mesh = UsdGeom.Mesh.Define(stage, "/World/House/Dock/tag")
    # Seen from the room, looking at the wall: +y is to the right, +z is up.
    mesh.CreatePointsAttr([(cx, cy - half, cz - half), (cx, cy + half, cz - half), (cx, cy + half, cz + half), (cx, cy - half, cz + half)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([(1.0, 0.0, 0.0)] * 4)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateExtentAttr([(cx, cy - half, cz - half), (cx, cy + half, cz + half)])
    st = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])

    base = "/World/Looks/dock_tag"
    mat = UsdShade.Material.Define(stage, base)
    reader = UsdShade.Shader.Define(stage, base + "/st")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    texture = UsdShade.Shader.Define(stage, base + "/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("./" + hl.DOCK_TAG_TEXTURE))
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    )
    shader = UsdShade.Shader.Define(stage, base + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)  # matte: no glare on the code
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)


def make_floor_physics_material(stage):
    """High-friction physics material so the wheels do not slip on the floor."""
    mat = UsdShade.Material.Define(stage, "/World/Looks/floor_physics")
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(1.0)
    api.CreateDynamicFrictionAttr(1.0)
    api.CreateRestitutionAttr(0.0)
    return mat


def textured_material(stage, texture, cache):
    """A material showing a picture from sim/assets (relative path), repeated as the UVs say."""
    if texture in cache:
        return cache[texture]
    base = "/World/Looks/tex_" + os.path.splitext(os.path.basename(texture))[0]
    mat = UsdShade.Material.Define(stage, base)
    reader = UsdShade.Shader.Define(stage, base + "/st")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, base + "/Texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("./" + texture))
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.CreateOutput("result", Sdf.ValueTypeNames.Float2))
    shader = UsdShade.Shader.Define(stage, base + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    cache[texture] = mat
    return mat


def add_floor_surface(stage, shape, look, cache):
    """A visible floor drawn as its top surface with a picture on it (UVs in metres / tile size)."""
    texture, metres = look
    path = f"/World/House/{shape.group}/{shape.name}"
    wx0, wy0 = hl.to_world(shape.x0, shape.y0)
    wx1, wy1 = hl.to_world(shape.x1, shape.y1)
    z = shape.z1
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([(wx0, wy0, z), (wx1, wy0, z), (wx1, wy1, z), (wx0, wy1, z)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([(0.0, 0.0, 1.0)] * 4)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateExtentAttr([(wx0, wy0, z), (wx1, wy1, z)])
    st = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    # in the house frame, so the pattern lines up across a doorway from one room into the next
    st.Set([(x / metres, y / metres) for x, y in ((shape.x0, shape.y0), (shape.x1, shape.y0), (shape.x1, shape.y1), (shape.x0, shape.y1))])
    prim = mesh.GetPrim()
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(textured_material(stage, texture, cache))
    if shape.floor_type:
        prim.CreateAttribute("vacuum:floor_type", Sdf.ValueTypeNames.Token).Set(shape.floor_type)
    return prim


def add_ceiling_surface(stage, shape, mats):
    """A ceiling seen only from below: one face pointing down, not double-sided, so a camera above
    the home (the overview, the window's view) looks straight through it into the rooms."""
    path = f"/World/House/{shape.group}/{shape.name}"
    wx0, wy0 = hl.to_world(shape.x0, shape.y0)
    wx1, wy1 = hl.to_world(shape.x1, shape.y1)
    z = shape.z0
    mesh = UsdGeom.Mesh.Define(stage, path)
    # clockwise seen from above = counter-clockwise seen from below: the face points down
    mesh.CreatePointsAttr([(wx0, wy0, z), (wx0, wy1, z), (wx1, wy1, z), (wx1, wy0, z)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([(0.0, 0.0, -1.0)] * 4)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateDoubleSidedAttr(False)
    # RTX ignores doubleSided; it culls the back of prims marked singleSided (with
    # /rtx/hydra/faceCulling/enabled, set by run_sim.py and capture_preview.py)
    mesh.GetPrim().CreateAttribute("singleSided", Sdf.ValueTypeNames.Bool).Set(True)
    mesh.CreateExtentAttr([(wx0, wy0, z), (wx1, wy1, z)])
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mats[shape.material])
    return mesh.GetPrim()


def add_shape(stage, shape, mats, floor_physics, textures=None):
    """Author one Box or Cylinder from the layout as a USD prim in the world frame."""
    path = f"/World/House/{shape.group}/{shape.name}"
    look = hl.floor_look(shape.name) if shape.group == "Floor" and shape.visible and hasattr(hl, "floor_look") else None
    if look is not None and not shape.collide:
        return add_floor_surface(stage, shape, look, textures if textures is not None else {})
    if shape.group == "Ceiling" and isinstance(shape, hl.Box):
        return add_ceiling_surface(stage, shape, mats)
    if isinstance(shape, hl.Box):
        wx0, wy0 = hl.to_world(shape.x0, shape.y0)
        wx1, wy1 = hl.to_world(shape.x1, shape.y1)
        size = (wx1 - wx0, wy1 - wy0, shape.z1 - shape.z0)
        center = ((wx0 + wx1) / 2.0, (wy0 + wy1) / 2.0, (shape.z0 + shape.z1) / 2.0)
        geom = UsdGeom.Cube.Define(stage, path)
        geom.CreateSizeAttr(1.0)
        geom.CreateExtentAttr([(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)])
        xform = UsdGeom.Xformable(geom.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*center))
        xform.AddScaleOp().Set(Gf.Vec3f(*size))
    else:
        wx, wy = hl.to_world(shape.x, shape.y)
        height = shape.z1 - shape.z0
        geom = UsdGeom.Cylinder.Define(stage, path)
        geom.CreateRadiusAttr(shape.radius)
        geom.CreateHeightAttr(height)
        geom.CreateAxisAttr("Z")
        r = shape.radius
        geom.CreateExtentAttr([(-r, -r, -height / 2.0), (r, r, height / 2.0)])
        UsdGeom.Xformable(geom.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(wx, wy, (shape.z0 + shape.z1) / 2.0))

    prim = geom.GetPrim()
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(mats[shape.material])
    if shape.collide:
        UsdPhysics.CollisionAPI.Apply(prim)
        if shape.group == "Floor":
            binding.Bind(floor_physics, UsdShade.Tokens.weakerThanDescendants, "physics")
    if getattr(shape, "floor_type", ""):
        prim.CreateAttribute("vacuum:floor_type", Sdf.ValueTypeNames.Token).Set(shape.floor_type)
    if not getattr(shape, "visible", True):
        # Collision-only geometry: "guide" purpose keeps it out of renders and sensor images.
        UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.guide)
    return prim


def add_labels(stage):
    """Names on the world's objects and rooms, where Isaac Sim's labelled camera pictures read them
    (UsdSemantics labels): taxonomy 'class' = the kind (chair, bed, floor, wall), 'name' = the
    object's own name, 'room' = the room it is in. Only worlds that name their things have them."""
    if not hasattr(hl, "things"):
        return 0

    def label(path, **taxonomies):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"cannot label {path}: no such prim")
        for taxonomy, value in taxonomies.items():
            UsdSemantics.LabelsAPI.Apply(prim, taxonomy).CreateLabelsAttr().Set([value])

    count = 0
    for thing in hl.things():
        for shape in thing.shapes:
            label(f"/World/House/{shape.group}/{shape.name}", **{"class": thing.kind, "name": thing.name, "room": thing.room})
            count += 1
    for room in hl.rooms():
        label(f"/World/House/Floor/{room.name}_floor", **{"class": "floor", "name": f"{room.name}_floor", "room": room.name})
        count += 1
    for wall in hl.walls():
        label(f"/World/House/{wall.group}/{wall.name}", **{"class": "stairs" if wall.name.startswith("stairs") else "wall"})
        count += 1
    for glass in hl.windows():
        label(f"/World/House/{glass.group}/{glass.name}", **{"class": "window"})
        count += 1
    for part in hl.ceiling():
        label(f"/World/House/{part.group}/{part.name}", **{"class": "lamp" if part.name.endswith("_shade") else "ceiling"})
        count += 1
    return count


# Each room's ceiling lamp, a sphere light (the environment variables are for tuning). 450000 with a
# 10 cm radius lights the robot's camera picture to a mean of about 117 of 255 (the house, lit by
# the sky, gives 147); 30000 gave 6.
LAMP_INTENSITY = float(os.environ.get("VACUUM_LAMP_INTENSITY", 450000.0))
LAMP_RADIUS = float(os.environ.get("VACUUM_LAMP_RADIUS", 0.10))


def add_lights(stage):
    """Sky dome plus one sun-like light, and a lamp under the ceiling of every room in worlds that
    have them (the house has no ceiling: the dome lights its rooms)."""
    UsdGeom.Scope.Define(stage, "/World/Lights")
    for name, _room, x, y in (hl.lamps() if hasattr(hl, "lamps") else []):
        wx, wy = hl.to_world(x, y)
        lamp = UsdLux.SphereLight.Define(stage, f"/World/Lights/{name}")
        lamp.CreateRadiusAttr(LAMP_RADIUS)
        lamp.CreateIntensityAttr(LAMP_INTENSITY)
        lamp.CreateColorAttr(Gf.Vec3f(1.0, 0.93, 0.82))  # warm white
        UsdGeom.Xformable(lamp.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(wx, wy, hl.WALL_H - hl.LAMP_DROP))
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(380.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))
    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
    sun.CreateIntensityAttr(1100.0)
    sun.CreateAngleAttr(1.5)
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(35.0, 25.0, 0.0))


def add_camera(stage, path, eye, target, up=(0.0, 0.0, 1.0), focal_length=18.0):
    """Add a camera at `eye` looking at `target`."""
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(focal_length)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 200.0))
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up))
    UsdGeom.Xformable(cam.GetPrim()).AddTransformOp().Set(view.GetInverse())


def add_physics_scene(stage):
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)


def add_robot(stage, assets_root):
    """Reference the Create 3 at the world origin (base_link sits on the floor at z = 0)."""
    stage_utils.add_reference_to_stage(assets_root + ROBOT_ASSET, ROBOT_PATH)
    prim = stage.GetPrimAtPath(ROBOT_PATH)
    if not prim.IsValid():
        raise RuntimeError(f"failed to reference robot asset at {ROBOT_PATH}")
    fix_caster(stage)
    # With the asset's default solver settings the resting robot reports a false 5 mm/s forward
    # velocity and a 0.007 rad/s pitch rate, and creeps 0.15 mm/s. More solver iterations for this
    # one articulation remove all three at no measurable cost.
    # As shipped, the chassis centre of mass is directly above the wheel axle. With only a front
    # caster, nothing then resists pitching backward: braking from reverse tipped the robot 5
    # degrees. A real Roomba carries its weight ahead of the axle and rests on its caster.
    chassis = stage.GetPrimAtPath(BASE_LINK_PATH)
    UsdPhysics.MassAPI.Apply(chassis).CreateCenterOfMassAttr(Gf.Vec3f(*CHASSIS_CENTER_OF_MASS))
    solver = PhysxSchema.PhysxArticulationAPI.Apply(prim)
    solver.CreateSolverPositionIterationCountAttr(128)
    solver.CreateSolverVelocityIterationCountAttr(16)
    # Known cost of the forward centre of mass: a robot standing still creeps backwards 2 mm a
    # minute. Motor inertia on the wheel joints (physxJoint:armature) stops the creep but makes
    # the robot under-turn (0.01 kg m^2: 174 degrees for a commanded 180), so it is not used.
    return prim


def fix_caster(stage):
    """Make the asset's fixed caster ball behave like a rolling caster.

    As shipped, the caster collider is a fixed sphere whose bottom is 2.25 mm lower than the
    wheels, so the robot rests nose-up by about 1 degree, and it drags sideways with full
    friction when turning. Both would hurt later: a tilted lidar plane and under-rotation.
    These overrides live in our world file; the NVIDIA asset itself is not modified.
    """
    caster = stage.GetPrimAtPath(CASTER_COLLIDER_PATH)
    if not caster.IsValid():
        raise RuntimeError(f"caster collider not found at {CASTER_COLLIDER_PATH}; the robot asset layout changed")
    sphere = UsdGeom.Sphere(caster)
    center_z = UsdGeom.XformCache().GetLocalToWorldTransform(caster).ExtractTranslation()[2]
    wheel_bottom_z = WHEEL_CENTER_Z - hl.WHEEL_RADIUS
    radius = center_z - wheel_bottom_z
    if not 0.004 < radius < 0.012:
        raise RuntimeError(f"unexpected caster geometry (center z={center_z:.4f}); refusing to guess a radius")
    sphere.GetRadiusAttr().Set(radius)
    sphere.GetExtentAttr().Set([(-radius, -radius, -radius), (radius, radius, radius)])

    mat = UsdShade.Material.Define(stage, "/World/Looks/caster_physics")
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(0.0)
    api.CreateDynamicFrictionAttr(0.0)
    api.CreateRestitutionAttr(0.0)
    physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    physx_mat.CreateFrictionCombineModeAttr("min")
    physx_mat.CreateRestitutionCombineModeAttr("min")
    UsdShade.MaterialBindingAPI.Apply(caster).Bind(mat, UsdShade.Tokens.strongerThanDescendants, "physics")


def add_sensors(stage):
    """Author the sensor prims on the robot. All are children of base_link.

    Their fixed frames (laser, imu_link, camera_link, camera_optical_frame, cliff_*) are published
    on /tf_static by ros_publishers.py when run_sim.py starts.
    """
    import numpy as np
    from isaacsim.ros2.nodes import set_isaac_name_override
    from isaacsim.sensors.experimental.physics import IMU, Contact, Raycast
    from isaacsim.sensors.experimental.rtx import Lidar

    if hl.LIDAR_BACKEND == "rtx":
        # The stock config is blind inside 1 m and reaches 200 m; a vacuum needs the opposite.
        Lidar.create(
            path=LIDAR_PATH,
            config="Example_Rotary_2D",
            tick_rate=hl.LIDAR_RATE_HZ,
            translations=[list(hl.LIDAR_XYZ)],
            attributes={
                "omni:sensor:Core:nearRangeM": hl.LIDAR_MIN_RANGE,
                "omni:sensor:Core:farRangeM": hl.LIDAR_MAX_RANGE,
                "omni:sensor:Core:scanRateBaseHz": int(hl.LIDAR_RATE_HZ),
            },
        )
    else:
        # One physics raycast sensor with a fan of horizontal rays; ros_publishers.py turns its
        # depths into sensor_msgs/LaserScan. Beam i points at angle -pi + i * 2*pi/N (ROS convention).
        angles = -math.pi + np.arange(hl.LIDAR_BEAMS) * (2.0 * math.pi / hl.LIDAR_BEAMS)
        Raycast.create(
            path=LIDAR_PATH,
            translations=[list(hl.LIDAR_XYZ)],
            min_range=hl.LIDAR_MIN_RANGE,
            max_range=hl.LIDAR_MAX_RANGE,
            ray_origins=np.zeros((hl.LIDAR_BEAMS, 3)),
            ray_directions=np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1),
            output_frame="SENSOR",
            report_hit_prim_paths=False,
        )
    # A short visible turret under the scan plane (visual only, must stay below the beams).
    turret = UsdGeom.Cylinder.Define(stage, BASE_LINK_PATH + "/lidar_turret")
    turret_h = hl.LIDAR_XYZ[2] - 0.008 - 0.0925
    turret.CreateRadiusAttr(0.035)
    turret.CreateHeightAttr(turret_h)
    turret.CreateAxisAttr("Z")
    turret.CreateExtentAttr([(-0.035, -0.035, -turret_h / 2), (0.035, 0.035, turret_h / 2)])
    turret.CreateDisplayColorAttr([Gf.Vec3f(0.12, 0.12, 0.13)])
    UsdGeom.Xformable(turret.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(hl.LIDAR_XYZ[0], hl.LIDAR_XYZ[1], 0.0925 + turret_h / 2))

    # Forward RGB-D camera: camera_link has ROS body axes (x forward); the Camera prim under it
    # uses USD camera axes (looks along -z, +y up) and is published as the optical frame.
    link = UsdGeom.Xform.Define(stage, CAMERA_LINK_PATH)
    link.AddTranslateOp().Set(Gf.Vec3d(*hl.CAMERA_XYZ))
    cam = UsdGeom.Camera.Define(stage, CAMERA_PATH)
    aperture = 20.955
    cam.CreateHorizontalApertureAttr(aperture)
    cam.CreateVerticalApertureAttr(aperture * hl.CAMERA_HEIGHT / hl.CAMERA_WIDTH)
    cam.CreateFocalLengthAttr(aperture / (2.0 * math.tan(math.radians(hl.CAMERA_HFOV_DEG) / 2.0)))
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.02, 30.0))
    look = Gf.Matrix4d().SetLookAt(Gf.Vec3d(0, 0, 0), Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 0, 1))
    UsdGeom.Xformable(cam.GetPrim()).AddTransformOp().Set(look.GetInverse())
    set_isaac_name_override(CAMERA_PATH, "camera_optical_frame")

    # IMU on the asset's imu_link mount frame.
    IMU.create(path=IMU_PATH, translations=[[0.0, 0.0, 0.0]])

    # Bumper: one contact sensor for the whole chassis; run_sim.py turns contact points into zones.
    Contact.create(path=BUMPER_PATH, min_threshold=0.0, max_threshold=100000.0, radius=-1.0)

    # Cliff: one raycast sensor with four downward rays at iRobot's sensor positions.
    names = list(hl.CLIFF_SENSORS)
    Raycast.create(
        path=CLIFF_PATH,
        translations=[[0.0, 0.0, 0.0]],
        min_range=0.0,
        max_range=hl.CLIFF_MAX_RANGE,
        ray_origins=np.array([[hl.CLIFF_SENSORS[n][0], hl.CLIFF_SENSORS[n][1], hl.CLIFF_RAY_Z] for n in names]),
        ray_directions=np.array([[0.0, 0.0, -1.0]] * len(names)),
        output_frame="SENSOR",
        report_hit_prim_paths=True,
    )


def add_sensor_graphs():
    """ROS 2 publisher graphs for the camera (and the RTX lidar when that backend is selected)."""
    from isaacsim.ros2.nodes import (
        Ros2CameraGraphConfig,
        Ros2RtxLidarGraphConfig,
        create_ros2_camera_graph,
        create_ros2_rtx_lidar_graph,
    )

    if hl.LIDAR_BACKEND == "rtx":
        create_ros2_rtx_lidar_graph(
            Ros2RtxLidarGraphConfig(
                graph_path="/World/Graphs/ROS_Lidar",
                lidar_prim=LIDAR_PATH,
                frame_id="laser",
                publish_laser_scan=True,
                laser_scan_topic="scan",
            )
        )
        # The lidar needs a render product, but its size is irrelevant; keep it tiny.
        lidar_graph = "/World/Graphs/ROS_Lidar"
        og.Controller.edit(
            og.get_graph_by_path(lidar_graph),
            {
                og.Controller.Keys.SET_VALUES: [
                    (f"{lidar_graph}/RenderProduct.inputs:width", 64),
                    (f"{lidar_graph}/RenderProduct.inputs:height", 64),
                ]
            },
        )

    create_ros2_camera_graph(
        Ros2CameraGraphConfig(
            graph_path="/World/Graphs/ROS_Camera",
            camera_prim=CAMERA_PATH,
            frame_id="camera_optical_frame",
            camera_info_topic="camera/camera_info",
            publish_rgb=True,
            rgb_topic="camera/image_raw",
            publish_depth=True,
            depth_topic="camera/depth/image_raw",
        )
    )
    # The builder defaults to 1280x720 on every frame; a vacuum camera needs far less.
    # Values must go through Controller.edit to be saved into the USD (Controller.set is runtime only).
    skip = max(0, int(round(SIM_RATE_HZ / hl.CAMERA_RATE_HZ)) - 1)
    cam_graph = "/World/Graphs/ROS_Camera"
    og.Controller.edit(
        og.get_graph_by_path(cam_graph),
        {
            og.Controller.Keys.SET_VALUES: [
                (f"{cam_graph}/RenderProduct.inputs:width", hl.CAMERA_WIDTH),
                (f"{cam_graph}/RenderProduct.inputs:height", hl.CAMERA_HEIGHT),
                (f"{cam_graph}/CameraInfoPublish.inputs:frameSkipCount", skip),
                (f"{cam_graph}/RGBPublish.inputs:frameSkipCount", skip),
                (f"{cam_graph}/DepthPublish.inputs:frameSkipCount", skip),
            ]
        },
    )

def add_ros_graphs():
    """Clock graph and the wheel drive graph, both ticked by playback."""
    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": "/World/Graphs/ROS_Clock", "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ],
            keys.SET_VALUES: [("PublishClock.inputs:topicName", "/clock")],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
            ],
        },
    )

    # Odometry, its TF and the IMU are published by ros_publishers.py, which can add noise.
    robot = [usdrt.Sdf.Path(ROBOT_PATH)]
    og.Controller.edit(
        {"graph_path": "/World/Graphs/ROS_Drive", "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                # velocity commands in
                ("SubscribeTwist", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
                ("BreakLinVel", "omni.graph.nodes.BreakVector3"),
                ("BreakAngVel", "omni.graph.nodes.BreakVector3"),
                ("DiffController", "isaacsim.robot.wheeled_robots.DifferentialController"),
                ("ArtController", "isaacsim.core.nodes.IsaacArticulationController"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                ("OnPlaybackTick.outputs:tick", "SubscribeTwist.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "ArtController.inputs:execIn"),
                # Tick the controller every frame (not only when a Twist arrives) so its
                # acceleration limits ramp the wheel speeds smoothly against real elapsed time.
                ("OnPlaybackTick.outputs:tick", "DiffController.inputs:execIn"),
                ("OnPlaybackTick.outputs:deltaSeconds", "DiffController.inputs:dt"),
                ("SubscribeTwist.outputs:linearVelocity", "BreakLinVel.inputs:tuple"),
                ("BreakLinVel.outputs:x", "DiffController.inputs:linearVelocity"),
                ("SubscribeTwist.outputs:angularVelocity", "BreakAngVel.inputs:tuple"),
                ("BreakAngVel.outputs:z", "DiffController.inputs:angularVelocity"),
                ("DiffController.outputs:velocityCommand", "ArtController.inputs:velocityCommand"),
            ],
            keys.SET_VALUES: [
                ("PublishJointState.inputs:topicName", "joint_states"),
                ("PublishJointState.inputs:targetPrim", robot),
                ("SubscribeTwist.inputs:topicName", hl.WHEEL_CMD_TOPIC),
                # Newest command only. This node reads one message per frame; with the default
                # queue of 10, a publisher even slightly faster than the frame rate fills the
                # queue and every command then reaches the wheels ten frames late. That delay
                # (6 cm at full speed) was enough to take the robot over the stairwell edge.
                ("SubscribeTwist.inputs:queueSize", 1),
                ("DiffController.inputs:wheelRadius", hl.WHEEL_RADIUS),
                ("DiffController.inputs:wheelDistance", hl.WHEEL_SEPARATION),
                ("DiffController.inputs:maxLinearSpeed", hl.MAX_LINEAR_SPEED),
                ("DiffController.inputs:maxAngularSpeed", hl.MAX_ANGULAR_SPEED),
                ("DiffController.inputs:maxAcceleration", hl.MAX_LINEAR_ACCEL),
                ("DiffController.inputs:maxDeceleration", hl.MAX_LINEAR_DECEL),
                ("DiffController.inputs:maxAngularAcceleration", hl.MAX_ANGULAR_ACCEL),
                ("ArtController.inputs:targetPrim", robot),
                ("ArtController.inputs:jointNames", [LEFT_WHEEL_JOINT, RIGHT_WHEEL_JOINT]),
            ],
        },
    )


def main():
    app_utils.enable_extension("isaacsim.ros2.bridge")
    simulation_app.update()

    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("could not find the Isaac Sim assets root")

    stage_utils.create_new_stage()
    simulation_app.update()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    mats = make_materials(stage)
    floor_physics = make_floor_physics_material(stage)
    shapes = hl.all_shapes()
    for group in ("Floor", "Walls", "Windows", "Ceiling", "Furniture", "Doors", "Dock"):
        if any(s.group == group for s in shapes):
            UsdGeom.Xform.Define(stage, f"/World/House/{group}")
    textures = {}
    for shape in shapes:
        add_shape(stage, shape, mats, floor_physics, textures)
    labelled = add_labels(stage)
    texture = os.path.join(os.path.dirname(os.path.abspath(args.out)), hl.DOCK_TAG_TEXTURE)
    if not os.path.isfile(texture):
        raise RuntimeError(f"{texture} is missing; run: python3 sim/scripts/make_apriltag.py")
    add_dock_tag(stage)

    add_lights(stage)
    add_physics_scene(stage)

    # Cameras for previews: an angled overview and a straight top-down view of the whole world.
    # Placed for the house's 8.8 x 4.7 m outline and moved back in proportion for a bigger world.
    fx0, fy0, fx1, fy1 = hl.FOOTPRINT
    hx0, hy0 = hl.to_world(fx0, fy0)
    hx1, hy1 = hl.to_world(fx1, fy1)
    cx, cy = round((hx0 + hx1) / 2.0, 6), round((hy0 + hy1) / 2.0, 6)
    k = max(1.0, (fx1 - fx0) / 8.8, (fy1 - fy0) / 4.7)
    UsdGeom.Scope.Define(stage, "/World/Cameras")
    add_camera(stage, "/World/Cameras/Overview", (cx, cy - 4.6 * k, 11.5 * k), (cx, cy - 0.15 * k, 0.0))
    add_camera(stage, "/World/Cameras/TopDown", (cx, cy, 13.0 * k), (cx, cy, 0.0), up=(0.0, 1.0, 0.0))

    add_robot(stage, assets_root)
    simulation_app.update()
    add_sensors(stage)
    simulation_app.update()
    add_ros_graphs()
    add_sensor_graphs()
    simulation_app.update()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if not stage_utils.save_stage(args.out):
        raise RuntimeError(f"failed to save stage to {args.out}")
    print(f"BUILD_OK world={hl.WORLD} shapes={len(shapes)} labelled={labelled} robot={ROBOT_PATH} out={args.out}")


try:
    main()
except Exception as exc:  # report and fail loudly; a half-built world must not look like success
    import traceback

    traceback.print_exc()
    print(f"BUILD_FAILED {exc}", flush=True)
    os._exit(1)  # simulation_app.close() would end the process with exit code 0

simulation_app.close()
