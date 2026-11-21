"""Print the structure of a robot USD asset: prims, joints, drives, colliders, sizes.

Usage (from a shell WITHOUT ROS sourced):
    cd ~/isaacsim && ./python.sh <this file> [usd path or path relative to the Isaac assets root]

With no argument it inspects the bundled iRobot Create 3.
"""

import sys

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

CANDIDATES = [
    "/Isaac/Robots/iRobot/Create3/create_3.usd",
    "/Isaac/Robots_Multiphysics/iRobot/Create3/create_3/create_3.usda",
    "/Isaac/Robots/iRobot/Create3/create_3/create_3.usda",
]


def open_stage(arg):
    root = get_assets_root_path()
    print(f"ASSETS_ROOT {root}")
    paths = [arg] if arg else CANDIDATES
    for p in paths:
        url = p if ("://" in p or p.startswith("/home")) else root + p
        stage = Usd.Stage.Open(url)
        if stage:
            print(f"OPENED {url}")
            return stage
        print(f"not found: {url}")
    return None


def fmt(v):
    if v is None:
        return "None"
    try:
        return "(" + ", ".join(f"{x:.5f}" for x in v) + ")"
    except TypeError:
        return f"{v:.5f}" if isinstance(v, float) else str(v)


def main():
    stage = open_stage(sys.argv[1] if len(sys.argv) > 1 else None)
    if stage is None:
        print("RESULT no stage opened")
        return
    print(f"upAxis={UsdGeom.GetStageUpAxis(stage)} metersPerUnit={UsdGeom.GetStageMetersPerUnit(stage)}")
    dp = stage.GetDefaultPrim()
    print(f"defaultPrim={dp.GetPath() if dp else None}")

    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    xf = UsdGeom.XformCache()

    print("\n== PRIM TREE (depth <= 4) ==")
    for prim in stage.Traverse():
        depth = len(prim.GetPath().pathString.split("/")) - 1
        if depth > 4:
            continue
        apis = [a for a in prim.GetAppliedSchemas()]
        print(f"{'  ' * (depth - 1)}{prim.GetName()} [{prim.GetTypeName()}] {apis if apis else ''}")

    print("\n== ARTICULATION ROOTS ==")
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            print(prim.GetPath())

    print("\n== RIGID BODIES ==")
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            pos = xf.GetLocalToWorldTransform(prim).ExtractTranslation()
            rng = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
            mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get() if prim.HasAPI(UsdPhysics.MassAPI) else None
            print(
                f"{prim.GetPath()} pos={fmt(pos)} bbox_min={fmt(rng.GetMin())} "
                f"bbox_max={fmt(rng.GetMax())} size={fmt(rng.GetSize())} mass={mass}"
            )

    print("\n== JOINTS ==")
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        j = UsdPhysics.Joint(prim)
        b0 = [str(t) for t in j.GetBody0Rel().GetTargets()]
        b1 = [str(t) for t in j.GetBody1Rel().GetTargets()]
        axis = prim.GetAttribute("physics:axis").Get() if prim.GetAttribute("physics:axis") else None
        print(f"{prim.GetPath()} type={prim.GetTypeName()} axis={axis}")
        print(f"   body0={b0} body1={b1}")
        print(f"   localPos0={fmt(j.GetLocalPos0Attr().Get())} localPos1={fmt(j.GetLocalPos1Attr().Get())}")
        for attr in prim.GetAttributes():
            n = attr.GetName()
            if n.startswith("drive:") or "maxJointVelocity" in n or n.startswith("physics:lowerLimit") or n.startswith(
                "physics:upperLimit"
            ):
                print(f"   {n} = {attr.Get()}")

    print("\n== COLLIDERS ==")
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            approx = prim.GetAttribute("physics:approximation").Get() if prim.GetAttribute("physics:approximation") else None
            rng = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
            extra = ""
            for a in ("radius", "height", "size", "axis"):
                at = prim.GetAttribute(a)
                if at and at.Get() is not None:
                    extra += f" {a}={at.Get()}"
            print(f"{prim.GetPath()} [{prim.GetTypeName()}] approx={approx} size={fmt(rng.GetSize())}{extra}")

    print("\n== OVERALL BOUNDS ==")
    if dp:
        rng = bbox.ComputeWorldBound(dp).ComputeAlignedRange()
        print(f"min={fmt(rng.GetMin())} max={fmt(rng.GetMax())} size={fmt(rng.GetSize())}")
    print("RESULT done")


try:
    main()
finally:
    simulation_app.close()
