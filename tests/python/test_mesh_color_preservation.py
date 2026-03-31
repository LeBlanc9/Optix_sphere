import os
import tempfile
from pathlib import Path
import optix_sphere as osg

def test_mesh_color_preservation():
    # 1. 创建临时的 OBJ 和 MTL 文件，包含特定的颜色
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        obj_path = tmp_path / "test.obj"
        mtl_path = tmp_path / "test.mtl"
        
        # 定义测试颜色
        kd_val = "0.123456 0.456789 0.789123"
        ka_val = "0.010000 0.020000 0.030000"
        ks_val = "0.900000 0.800000 0.700000"
        
        # 写入 MTL
        with open(mtl_path, "w") as f:
            f.write(f"newmtl test_material\n")
            f.write(f"Ka {ka_val}\n")
            f.write(f"Kd {kd_val}\n")
            f.write(f"Ks {ks_val}\n")
            f.write("Ns 10.0\n")
            f.write("d 1.0\n")
            
        # 写入 OBJ
        with open(obj_path, "w") as f:
            f.write(f"mtllib test.mtl\n")
            f.write("v 0 0 0\n")
            f.write("v 1 0 0\n")
            f.write("v 0 1 0\n")
            f.write("usemtl test_material\n")
            f.write("f 1 2 3\n")
            
        print(f"--- Created temporary test files in {tmp_dir} ---")
        
        # 2. 加载 Mesh
        mesh = osg.Mesh.from_obj(str(obj_path))
        print(f"Loaded mesh with {mesh.get_material_count()} materials.")
        assert "test_material" in mesh.material_names
        
        # 3. 导出到新文件
        out_obj = tmp_path / "exported.obj"
        out_mtl = tmp_path / "exported.mtl"
        mesh.to_obj(str(out_obj))
        
        # 4. 验证导出的 MTL 文件内容
        assert out_mtl.exists()
        with open(out_mtl, "r") as f:
            content = f.read()
            print("\n--- Exported MTL Content ---")
            print(content)
            
            # 验证颜色数值是否存在
            # 注意：由于浮点数精度或格式化原因，我们检查核心数值
            assert "newmtl test_material" in content
            assert "Kd 0.123456 0.456789 0.789123" in content
            assert "Ka 0.010000 0.020000 0.030000" in content
            assert "Ks 0.900000 0.800000 0.700000" in content
            
        print("\n✅ Test Passed: Material colors were correctly preserved!")

if __name__ == "__main__":
    try:
        test_mesh_color_preservation()
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
