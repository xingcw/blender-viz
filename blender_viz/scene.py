"""Blender-side scene builder. Invoked by ``python -m blender_viz``.

Keep this file importable only inside Blender: the public CLI and data loaders do
not require bpy, which makes rollout validation possible on compute nodes.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector, Quaternion


SPEC = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text())


def mat(name, color, metallic=0.0, roughness=0.35, emission=None, strength=0.0, alpha=1.0):
    material = bpy.data.materials.new(name)
    material.diffuse_color = (*color, alpha)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if emission is not None:
        socket = bsdf.inputs.get("Emission Color") or bsdf.inputs.get("Emission")
        socket.default_value = (*emission, 1)
        bsdf.inputs["Emission Strength"].default_value = strength
    if alpha < 1:
        bsdf.inputs["Alpha"].default_value = alpha
        if hasattr(material, "surface_render_method"):
            material.surface_render_method = "DITHERED"
        else:
            material.blend_method = "BLEND"
    return material


M = {}
GATE_PROTOTYPES = None


def materials():
    M.update({
        "carbon": mat("Carbon fiber", (0.012, .016, .025), metallic=.72, roughness=.22),
        "metal": mat("Anodized metal", (.06, .085, .13), metallic=.9, roughness=.18),
        "white": mat("Gate ceramic", (.55, .65, .78), metallic=.35, roughness=.23),
        "cyan": mat("Electric cyan", (.005, .32, .72), emission=(.005, .35, 1), strength=10),
        "blue": mat("Velocity blue", (.005, .08, .8), emission=(.005, .06, 1), strength=14),
        "violet": mat("Velocity violet", (.35, .015, .75), emission=(.45, .01, 1), strength=14),
        "pink": mat("Velocity magenta", (1, .015, .18), emission=(1, .005, .08), strength=16),
        "amber": mat("Gate amber", (1, .12, .008), emission=(1, .05, .002), strength=12),
        "glass": mat("Propeller glass", (.015, .15, .32), metallic=.15, roughness=.08, alpha=.42),
        "ground": mat("Arena floor", (.008, .012, .022), metallic=.25, roughness=.28),
    })


def cube(name, location, scale, material, bevel=0.0, rotation=None):
    bpy.ops.mesh.primitive_cube_add(location=location, rotation=rotation or (0, 0, 0))
    obj = bpy.context.object; obj.name = name; obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    if bevel:
        mod = obj.modifiers.new("Soft machined edges", "BEVEL"); mod.width = bevel; mod.segments = 3
    return obj


def uv_sphere(name, location, radius, material):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, location=location, radius=radius)
    obj = bpy.context.object; obj.name = name; obj.data.materials.append(material)
    bpy.ops.object.shade_smooth(); return obj


def cylinder_between(name, a, b, radius, material, vertices=16):
    a, b = Vector(a), Vector(b); delta = b - a
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=delta.length, location=(a + b) / 2)
    obj = bpy.context.object; obj.name = name; obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = delta.to_track_quat("Z", "Y"); obj.data.materials.append(material)
    bpy.ops.object.shade_smooth(); return obj


def curve(name, points, material, bevel=.025, cyclic=False):
    data = bpy.data.curves.new(name, "CURVE"); data.dimensions = "3D"; data.resolution_u = 3
    data.bevel_depth = bevel; data.bevel_resolution = 5
    spline = data.splines.new("NURBS"); spline.points.add(len(points) - 1)
    for p, co in zip(spline.points, points): p.co = (*co, 1)
    spline.order_u = min(4, len(points)); spline.use_endpoint_u = not cyclic; spline.use_cyclic_u = cyclic
    obj = bpy.data.objects.new(name, data); bpy.context.collection.objects.link(obj); obj.data.materials.append(material)
    return obj


def create_gate(gate, index):
    """Instance UZH Flightmare's autonomous-racing gate asset."""
    global GATE_PROTOTYPES
    root = bpy.data.objects.new(f"Gate {index + 1:02d}", None); bpy.context.collection.objects.link(root)
    root.location = gate["position"]; root.rotation_euler = gate["rotation"]
    if GATE_PROTOTYPES is None:
        asset_path = SPEC.get("gate_asset")
        if not asset_path or not Path(asset_path).exists():
            raise RuntimeError(f"Flightmare gate asset not found: {asset_path}")
        with bpy.data.libraries.load(asset_path, link=False) as (source, target):
            target.objects = [name for name in ("Truss", "CoverB", "CoverF") if name in source.objects]
        GATE_PROTOTYPES = target.objects
        texture_path = SPEC.get("gate_texture")
        if texture_path and Path(texture_path).exists():
            cover = bpy.data.materials.new("Unbranded printed gate fabric"); cover.use_nodes = True
            nodes = cover.node_tree.nodes; links = cover.node_tree.links
            bsdf = nodes.get("Principled BSDF"); bsdf.inputs["Roughness"].default_value = .62
            image_node = nodes.new("ShaderNodeTexImage"); image_node.image = bpy.data.images.load(texture_path, check_existing=True)
            links.new(image_node.outputs["Color"], bsdf.inputs["Base Color"])
            for obj in GATE_PROTOTYPES:
                if obj and obj.name.startswith("Cover"):
                    obj.data.materials.clear(); obj.data.materials.append(cover)
    holder = bpy.data.objects.new(f"Gate {index+1} asset transform", None); bpy.context.collection.objects.link(holder)
    holder.parent = root
    holder.rotation_euler.z = math.pi / 2
    # Upstream opening is 2.438 m; preserve the environment's exact 1 m opening.
    holder.scale = (1/2.438,)*3
    for prototype in GATE_PROTOTYPES:
        obj = prototype.copy(); obj.data = prototype.data.copy(); obj.name = f"Gate {index+1} {prototype.name}"
        bpy.context.collection.objects.link(obj); obj.parent = holder
    return root


def create_drone(name="Racing drone", scale=1.0):
    root = bpy.data.objects.new(name, None); bpy.context.collection.objects.link(root)
    root.rotation_mode = "QUATERNION"
    body = cube("Flight controller", (0, 0, 0), (.12, .09, .035), M["carbon"], .025); body.parent = root
    canopy = uv_sphere("Camera canopy", (.07, 0, .045), .07, M["metal"]); canopy.scale = (1.25, .8, .65); canopy.parent = root
    # X-frame arms, motors, luminous prop discs.
    for i, (x, y) in enumerate(((.18,.18),(.18,-.18),(-.18,.18),(-.18,-.18))):
        arm = cylinder_between("Carbon arm", (0, 0, 0), (x, y, 0), .018, M["carbon"], 12); arm.parent = root
        motor = cylinder_between("Motor", (x,y,-.012), (x,y,.055), .032, M["metal"], 20); motor.parent = root
        bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=.105, depth=.004, location=(x,y,.062))
        prop = bpy.context.object; prop.name = f"Motion-blurred propeller {i+1}"; prop.data.materials.append(M["glass"]); prop.parent = root
        hub = uv_sphere("Prop hub", (x,y,.067), .022, M["cyan"]); hub.parent = root
    # forward-facing camera lens and rear status light
    camera = cylinder_between("FPV camera", (.125,0,.035), (.17,0,.035), .03, M["carbon"], 24); camera.parent = root
    lens = uv_sphere("FPV lens", (.174,0,.035), .022, M["cyan"]); lens.scale.x=.35; lens.parent=root
    tail = uv_sphere("Tail light", (-.13,0,.035), .025, M["pink"]); tail.parent=root
    root.scale = (scale, scale, scale)
    return root


def compute_speeds(points):
    supplied = SPEC["trajectory"].get("speeds")
    if supplied: return [float(v) for v in supplied]
    speeds = [0.0]
    for a, b in zip(points, points[1:]): speeds.append((Vector(b)-Vector(a)).length)
    if len(speeds) > 2: speeds[0] = speeds[1]
    return speeds


def create_trail(points):
    # Soft cyan underglow plus short quantized speed-color segments.
    curve("Trajectory aura", points, M["cyan"], .045)
    speeds = compute_speeds(points); lo, hi = min(speeds), max(speeds); span = max(hi-lo, 1e-6)
    bins = ("blue", "violet", "pink")
    start, current = 0, None
    for i in range(len(points)-1):
        key = bins[min(2, int(3 * (speeds[i]-lo) / span))]
        if current is None: current = key
        if key != current or i == len(points)-2:
            end = i + 1 if i == len(points)-2 else i
            if end > start: curve(f"Speed trail {start:04d}", points[start:end+1], M[current], .023)
            start, current = i, key
    # Direction chevrons along the line.
    stride = max(8, len(points)//14)
    for i in range(stride, len(points)-1, stride):
        tangent = Vector(points[min(i+2, len(points)-1)]) - Vector(points[max(0, i-2)])
        bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=.09, radius2=0, depth=.24, location=points[i])
        arrow = bpy.context.object; arrow.name="Direction chevron"; arrow.rotation_mode="QUATERNION"
        arrow.rotation_quaternion=tangent.to_track_quat("Z","Y"); arrow.data.materials.append(M["amber"])


def tangent_quaternion(points, i):
    tangent = Vector(points[min(i+2, len(points)-1)]) - Vector(points[max(0, i-2)])
    q = tangent.to_track_quat("X", "Z")
    # subtle bank derived from turn curvature
    if 2 <= i < len(points)-2:
        a=(Vector(points[i])-Vector(points[i-2])).normalized(); b=(Vector(points[i+2])-Vector(points[i])).normalized()
        bank=max(-.65,min(.65,a.cross(b).z*1.8)); q = q @ Quaternion((1,0,0), bank)
    return q


def animate_drone(drone, points):
    n = len(points); frames = max(72, min(360, n)); bpy.context.scene.frame_end = frames
    quats = SPEC["trajectory"].get("quaternions")
    for frame in range(1, frames+1):
        i = round((frame-1)/(frames-1)*(n-1)); drone.location=points[i]
        drone.rotation_quaternion = Quaternion(quats[i]) if quats else tangent_quaternion(points, i)
        drone.keyframe_insert("location", frame=frame); drone.keyframe_insert("rotation_quaternion", frame=frame)
    if drone.animation_data and hasattr(drone.animation_data.action, "fcurves"):
        for curve_ in drone.animation_data.action.fcurves:
            for key in curve_.keyframe_points: key.interpolation="LINEAR"


def setup_world(points):
    scene=bpy.context.scene
    try:
        scene.render.engine=SPEC["engine"]
    except TypeError:
        # Eevee Next kept the legacy enum name in early Blender 4.x builds.
        if SPEC["engine"] == "BLENDER_EEVEE_NEXT": scene.render.engine="BLENDER_EEVEE"
        else: raise
    if scene.render.engine == "CYCLES": scene.cycles.samples=SPEC["samples"]
    scene.render.image_settings.file_format="PNG"
    scene.render.resolution_x=SPEC["width"]; scene.render.resolution_y=SPEC["height"]; scene.render.resolution_percentage=100
    scene.render.film_transparent=SPEC.get("transparent", True); scene.render.image_settings.color_mode="RGBA"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.world.use_nodes=True; bg=scene.world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value=(.002,.004,.012,1); bg.inputs["Strength"].default_value=.12
    # Ground and concentric arena markings.
    xs=[p[0] for p in points]; ys=[p[1] for p in points]; extent=max(max(xs)-min(xs),max(ys)-min(ys),5)
    if not SPEC.get("transparent", True):
        floor=cube("Reflective arena floor", ((max(xs)+min(xs))/2,(max(ys)+min(ys))/2,-.12), (extent*.9,extent*.9,.08), M["ground"], .08)
        for radius in (extent*.38, extent*.62, extent*.82):
            ring=[]
            for i in range(97):
                a=2*math.pi*i/96; ring.append((floor.location.x+radius*math.cos(a),floor.location.y+radius*math.sin(a),-.025))
            curve("Arena light ring",ring,M["cyan"],.009,True)
    # Area key lights and low colored rim lights.
    center=Vector((sum(xs)/len(xs),sum(ys)/len(ys),sum(p[2] for p in points)/len(points)))
    for idx,(loc,color,energy,size) in enumerate((((center.x-extent,center.y-extent,7),(0.12,.3,1),1300,6),
                                                  ((center.x+extent,center.y+extent,5),(1,.03,.18),900,4))):
        data=bpy.data.lights.new(f"Arena light {idx}","AREA"); data.energy=energy; data.color=color; data.shape="DISK"; data.size=size
        obj=bpy.data.objects.new(data.name,data); bpy.context.collection.objects.link(obj); obj.location=loc
        obj.rotation_euler=(Vector((0,0,-1)).rotation_difference((center-obj.location).normalized())).to_euler()
    # Blender <=4.x exposes the compositor through Scene.node_tree. Blender 5
    # moved this to the new compositor asset API; emissive materials still
    # render correctly there, so keep the scene portable and omit only FOG_GLOW.
    if hasattr(scene, "node_tree"):
        scene.use_nodes=True; nodes=scene.node_tree.nodes; links=scene.node_tree.links; nodes.clear()
        rl=nodes.new("CompositorNodeRLayers"); glare=nodes.new("CompositorNodeGlare"); glare.glare_type="FOG_GLOW"; glare.quality="HIGH"; glare.threshold=.7; glare.size=7
        comp=nodes.new("CompositorNodeComposite"); links.new(rl.outputs["Image"],glare.inputs["Image"]); links.new(glare.outputs["Image"],comp.inputs["Image"])


def camera_for(points, drone):
    scene=bpy.context.scene; pts=[Vector(p) for p in points]
    # Include the gate envelope in framing, not just the flight samples. This
    # keeps every gate, its supports, and the full trajectory visible.
    bounds = pts[:]
    for gate in SPEC["gates"]:
        p = Vector(gate["position"])
        bounds.extend((p + Vector((0, 0, 1.25)), p - Vector((0, 0, 1.25)),
                       p + Vector((0, 1.0, 0)), p - Vector((0, 1.0, 0))))
    mins = Vector(tuple(min(p[i] for p in bounds) for i in range(3)))
    maxs = Vector(tuple(max(p[i] for p in bounds) for i in range(3)))
    center = (mins + maxs) * .5
    center.z = max(.65, center.z)
    radius = max((p-center).length for p in bounds)
    data=bpy.data.cameras.new("Cinematic camera"); cam=bpy.data.objects.new("Cinematic camera",data); bpy.context.collection.objects.link(cam); scene.camera=cam
    data.lens=48
    mode=SPEC["camera"]
    if mode=="top":
        cam.location=(center.x,center.y,maxs.z + radius*2.65); data.lens=52
    else:
        # Three-quarter global course view. Distance is derived from the
        # bounding sphere with generous room for the 16:9 frame and gate legs.
        view_direction = Vector((-1.0, -1.35, .82)).normalized()
        cam.location = center + view_direction * radius * 4.15
    target=bpy.data.objects.new("Camera target",None); bpy.context.collection.objects.link(target); target.location=center
    constraint=cam.constraints.new("TRACK_TO"); constraint.target=drone if mode=="chase" else target; constraint.track_axis="TRACK_NEGATIVE_Z"; constraint.up_axis="UP_Y"
    if mode=="chase":
        cam.parent=drone; cam.location=(-1.15,0,.48); constraint.target=drone
    return cam


def main():
    bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.curves, bpy.data.meshes, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        pass
    materials(); points=SPEC["trajectory"]["positions"]
    setup_world(points); create_trail(points)
    for i,g in enumerate(SPEC["gates"]): create_gate(g,i)
    # An invisible path rig retains chase-camera animation without placing a
    # drone model in the visualization.
    flight_rig=bpy.data.objects.new("Invisible flight path rig",None); bpy.context.collection.objects.link(flight_rig)
    flight_rig.rotation_mode="QUATERNION"; animate_drone(flight_rig,points)
    camera_for(points,flight_rig)
    scene=bpy.context.scene; scene.render.filepath=SPEC["output"]; scene.render.fps=SPEC["fps"]
    if SPEC.get("blend"): bpy.ops.wm.save_as_mainfile(filepath=SPEC["blend"])
    if SPEC["open_ui"]: return
    if SPEC["animation"]:
        scene.render.image_settings.file_format="FFMPEG"; scene.render.ffmpeg.format="MPEG4"; scene.render.ffmpeg.codec="H264"; scene.render.ffmpeg.constant_rate_factor="HIGH"
        bpy.ops.render.render(animation=True)
    else:
        scene.frame_set(round(scene.frame_end*.62)); bpy.ops.render.render(write_still=True)


main()
