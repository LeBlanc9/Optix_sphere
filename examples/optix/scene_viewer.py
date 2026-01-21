import numpy as np
import optix_sphere as osg
from optix_sphere.visualization import Viewer
from pathlib import Path
import math


def main():
    print("=" * 70)
    print("Scene Transformation & Photon Batch Alignment Example")
    print("=" * 70)

    # ========================================================================
    # 1. Load Scene (TWO COPIES for comparison)
    # ========================================================================
    print("\n1️⃣  Loading integrating sphere scene...")

    mesh_path = (
        Path(__file__).parent.parent /
        "../assets/validations/port_thickness/integrating_sphere_25.4_10.obj"
    )

    # Load TWO copies - one for original, one for transformed
    scene_original = osg.Scene.from_obj(str(mesh_path))
    scene_transformed = osg.Scene.from_obj(str(mesh_path))

    print(f"   ✅ Loaded: {scene_original.get_vertex_count():,} vertices, {scene_original.get_triangle_count():,} triangles")

    min_b, max_b = scene_original.get_bounds()
    print(f"   Original bounds: ({min_b.x:.2f}, {min_b.y:.2f}, {min_b.z:.2f}) to ({max_b.x:.2f}, {max_b.y:.2f}, {max_b.z:.2f})")

    sphere_center_original = np.array([
        (min_b.x + max_b.x) / 2,
        (min_b.y + max_b.y) / 2,
        (min_b.z + max_b.z) / 2
    ])
    print(f"   Center: ({sphere_center_original[0]:.2f}, {sphere_center_original[1]:.2f}, {sphere_center_original[2]:.2f})")

    # ========================================================================
    # 2. Generate Photon Batch (simulating layered media output)
    # ========================================================================
    print("\n2️⃣  Generating photon batch (simulating layered media output)...")

    # Create a spot source as example input
    source = osg.SpotSource()
    source.center_position = osg.float3(0.0, 0.0, -50.0)  # Start 50mm below origin
    source.direction = osg.float3(0.0, 0.0, 1.0)  # Pointing up (+Z)
    angle_rad = math.radians(15)  # Spot half-angle
    source.disk_normal = osg.float3(math.sin(angle_rad), 0.0, math.cos(angle_rad))
    source.radius = 5.0

    photon_batch = osg.generate_photons(source, 5000)
    print(f"   ✅ Generated {photon_batch.size():,} photons")
    print(f"   Total weight: {photon_batch.total_weight():.6f}")

    # ========================================================================
    # 3. Check Initial Alignment
    # ========================================================================
    print("\n3️⃣  Checking initial alignment...")

    host_batch = photon_batch.to_host()
    positions = np.array([[p.x, p.y, p.z] for p in host_batch.positions])

    print(f"   Photon position range:")
    print(f"     X: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}]")
    print(f"     Y: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}]")
    print(f"     Z: [{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}]")

    # ========================================================================
    # 4. Transform Scene to Align with Photons (only scene_transformed)
    # ========================================================================
    print("\n4️⃣  Transforming scene to align with photon batch...")

    # Rotate scene 180 degrees around Y-axis
    print("\n   🔄 Rotating scene 180° around Y-axis...")
    scene_transformed.rotate_y(180.0)

    # Translate scene to photon batch location
    print("\n   📍 Translating scene to photon batch location...")
    photon_center = np.mean(positions, axis=0)

    min_b, max_b = scene_transformed.get_bounds()
    scene_center = np.array([
        (min_b.x + max_b.x) / 2,
        (min_b.y + max_b.y) / 2,
        (min_b.z + max_b.z) / 2
    ])

    offset = photon_center - scene_center
    # Offset photons to enter sphere from port (move sphere up so photons enter from bottom)
    offset[2] += 50.0

    scene_transformed.translate(osg.float3(offset[0], offset[1], offset[2]))

    min_b, max_b = scene_transformed.get_bounds()
    print(f"   ✅ Scene bounds after transform: ({min_b.x:.2f}, {min_b.y:.2f}, {min_b.z:.2f}) to ({max_b.x:.2f}, {max_b.y:.2f}, {max_b.z:.2f})")

    # ========================================================================
    # 5. Create Comparison Visualization (Before vs After) - NEW API!
    # ========================================================================
    print("\n5️⃣  Creating comparison visualization with new Viewer API...")

    # BEFORE visualization
    print("   Generating 'BEFORE' (original scene)...")
    viewer_before = Viewer(title="BEFORE Transformation")
    viewer_before.num_photons = 1000
    viewer_before.add(scene_original).add(photon_batch)
    viewer_before.save("sphere_before_transform.html")

    # AFTER visualization
    print("   Generating 'AFTER' (transformed scene)...")
    viewer_after = Viewer(title="AFTER Transformation (Rotated 180°)")
    viewer_after.num_photons = 1000
    viewer_after.add(scene_transformed).add(photon_batch)
    viewer_after.save("sphere_after_transform.html")

    print("\n✅ Visualizations saved!")
    print(f"   BEFORE (original): sphere_before_transform.html")
    print(f"   AFTER (rotated 180°): sphere_after_transform.html")
    print("\n   🟢 Green = photons inside sphere")
    print("   🔴 Red = photons outside sphere")

    # ========================================================================
    # 6. Summary & Next Steps
    # ========================================================================
    print("\n" + "=" * 70)
    print("📝 Summary")
    print("=" * 70)
    print(f"Scene: {scene_transformed.get_triangle_count():,} triangles")
    print(f"Photons: {photon_batch.size():,} (weight: {photon_batch.total_weight():.6f})")

    # Check final containment (using transformed scene)
    min_b, max_b = scene_transformed.get_bounds()
    sphere_center_final = np.array([
        (min_b.x + max_b.x) / 2,
        (min_b.y + max_b.y) / 2,
        (min_b.z + max_b.z) / 2
    ])
    sphere_radius = max(max_b.x - min_b.x, max_b.y - min_b.y, max_b.z - min_b.z) / 2

    from optix_sphere.visualization import check_photons_inside_sphere
    inside_mask, _ = check_photons_inside_sphere(positions, sphere_center_final, sphere_radius)

    if np.sum(inside_mask) == len(positions):
        print("\n✅ All photons are inside the sphere!")
    elif np.sum(inside_mask) > 0.9 * len(positions):
        print(f"\n⚠️  Most photons inside ({100*np.sum(inside_mask)/len(positions):.1f}%), but some are outside")
        print("   You may need to adjust the translation offset")
    else:
        print(f"\n❌ Many photons outside ({100*np.sum(~inside_mask)/len(positions):.1f}%)")
        print("   Recommended: Adjust scene.translate() offset or photon initial position")

    print("\n🚀 Next Steps:")
    print("   1. Fine-tune scene.translate() offset to ensure all photons enter")
    print("   2. Use the transformed scene in Simulator.build_scene()")
    print("   3. Run Monte Carlo simulation with sim.run(photon_batch, config)")
    print("=" * 70)


if __name__ == "__main__":
    main()
