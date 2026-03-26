from pathlib import Path
import tempfile

import optix_sphere as osg


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_OBJ = PROJECT_ROOT / "assets" / "R_25.4_1mm.obj"


def test_mesh_export_obj_and_mtl_roundtrip():
    mesh = osg.Mesh.from_obj(str(SOURCE_OBJ))

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        out_obj = tmp_path / "roundtrip.obj"

        # mtl_file_path is omitted on purpose to verify default behavior.
        mesh.to_obj(str(out_obj))

        out_mtl = tmp_path / "roundtrip.mtl"
        assert out_obj.exists(), f"OBJ not exported: {out_obj}"
        assert out_mtl.exists(), f"MTL not exported: {out_mtl}"

        mtl_text = out_mtl.read_text(encoding="utf-8")
        for mat_name in mesh.material_names:
            assert f"newmtl {mat_name}" in mtl_text

        loaded = osg.Mesh.from_obj(str(out_obj))

        assert loaded.get_triangle_count() == mesh.get_triangle_count()
        assert loaded.get_vertex_count() == mesh.get_vertex_count()
        assert loaded.get_material_count() == mesh.get_material_count()
        assert set(loaded.material_names) == set(mesh.material_names)


if __name__ == "__main__":
    test_mesh_export_obj_and_mtl_roundtrip()
    print("mesh export OBJ/MTL roundtrip test passed")
