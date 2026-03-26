import optix_sphere as osg
import time
from pathlib import Path
import pytest # If you have pytest, otherwise simple assert

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MESH_PATH = PROJECT_ROOT / "assets/R_25.4_1mm.obj"

def test_material_system():
    print("\n" + "="*60)
    print("🚀 Testing New Material System API")
    print("="*60)

    # 1. Initialize Simulator
    sim = osg.Simulator()
    scene = osg.Scene.from_obj(str(MESH_PATH))
    
    # Setup initial materials
    mats = {
        "wall_material": osg.material.lambertian(0.99),
        "detector_material": osg.material.detector(),
        "sample_material": osg.material.lambertian(0.0)
    }

    # 2. Build Scene (Dictionary Mapping)
    print("\n[Step 1] Building scene with complete mapping...")
    sim.build_scene(scene, mats)
    print(f"✅ Registered Materials: {sim.get_material_names()}")
    assert "wall_material" in sim.get_material_names()
    assert sim.get_material_count() == 3

    # 3. Test Property-based Update
    print("\n[Step 2] Testing Property-based Update...")
    wall = sim.get_material("wall_material")
    old_rho = wall.reflectance
    print(f"   Initial reflectance: {old_rho}")
    
    wall.reflectance = 0.50
    sim.sync_to_gpu()
    print(f"   Updated reflectance to: {wall.reflectance} (and synced to GPU)")
    assert wall.reflectance == 0.50

    # 4. Test Name-based set_material
    print("\n[Step 3] Testing Name-based set_material (Swap to Mixed)...")
    target_rho = 0.95
    mixed_mat = osg.material.mixed(0.7, 0.3, target_rho)
    sim.set_material("wall_material", mixed_mat, sync_to_gpu=True)
    
    new_wall = sim.get_material("wall_material")
    print(f"   New material type: {type(new_wall)}")
    print(f"   Reflectance: {new_wall.reflectance} (Target: {target_rho})")
    assert isinstance(new_wall, osg.MixedMaterial)
    # Use small epsilon for float comparison
    assert abs(new_wall.reflectance - target_rho) < 1e-6

    # 5. Test Missing Material (Warning + Default Absorber)
    print("\n[Step 4] Testing Missing Material (should trigger warning/fallback)...")
    # Only provide wall and sample, MISSING 'detector_material'
    incomplete_mats = {
        "wall_material": osg.material.lambertian(0.98),
        "sample_material": osg.material.lambertian(0.05)
    }
    
    # This should print a warning but not crash
    sim.build_scene(scene, incomplete_mats)
    print(f"   Final materials after build: {sim.get_material_names()}")
    
    # Verify that detector_material was auto-added (as an absorber)
    assert "detector_material" in sim.get_material_names()
    detector = sim.get_material("detector_material")
    print(f"   Auto-assigned material for detector: {type(detector)}")
    assert isinstance(detector, osg.DetectorMaterial) # Should use default factory for "detector_material"

    # 6. Test Setting Non-existent material (Should Raise Error)
    print("\n[Step 5] Testing non-existent material update (Should fail)...")
    try:
        sim.set_material("Imaginary_Wall", osg.material.lambertian(1.0))
        print("❌ Error: Should have raised a RuntimeError!")
    except RuntimeError as e:
        print(f"   ✅ Correctly caught expected error: {e}")

    # 7. Quick Simulation Test
    print("\n[Step 6] Running simulation with final setup...")
    source = osg.IsotropicPointSource()
    source.position = (0, 0, 0)
    sim.config.num_rays = 100000
    
    result = sim.run(source)
    print(f"   Simulation result: {result.irradiance:.6f} W/mm²")
    print("✅ All Material System tests passed!")

if __name__ == "__main__":
    test_material_system()
