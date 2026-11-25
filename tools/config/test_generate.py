"""Configuration codegen behavior, independent of any C++ build or GPU."""

import copy
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("config_generate", HERE / "generate.py")
GEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GEN)
SCHEMA = json.loads((HERE / "schema.json").read_text(encoding="utf-8"))


class GenerationTests(unittest.TestCase):
    def test_enum_names_match_shared_cpp_definitions(self):
        root = HERE.parents[1]
        shared = (root / "source/runtime/render/shaderheaders/shared/raster/SharedEnum.h").read_text(encoding="utf-8")
        for kind, members in re.findall(r"EnumParam\((\w+),\s*([^)]*)\)", shared):
            if kind in SCHEMA["enums"]:
                self.assertEqual([member.strip() for member in members.split(",")], SCHEMA["enums"][kind])
        for path, kind, sentinel in (
            ("source/runtime/render/shaderheaders/shared/ShaderParameters.h", "Render::EFinalColor", "EFC_NUM"),
            ("source/runtime/render/renderer/raster/RasterConfig.h", "EUpsampleMode", None),
            ("source/runtime/render/renderer/raytracing/RaytracingConfig.h", "EAnitiAliasMode", "EAA_Num"),
            ("source/runtime/render/renderer/raytracing/RaytracingConfig.h", "EOutputTexture", "EOT_Num"),
            ("source/runtime/render/renderer/raytracing/RaytracingConfig.h", "EJitter", "Num"),
        ):
            source = (root / path).read_text(encoding="utf-8")
            body = re.search(r"enum(?: class)? " + kind.split("::")[-1] + r"\s*\{([^}]+)\}", source).group(1)
            names = [member.strip().split("=")[0].strip() for member in body.split(",") if member.strip()]
            self.assertEqual([name for name in names if name != sentinel], SCHEMA["enums"][kind])

    def test_default_changes_do_not_change_public_declarations_or_loaders(self):
        base = GEN.generate(SCHEMA)
        for target, member, value, expected in (
            ("RasterConfig", "ssao_intensity", 2.25, "RasterConfigDefaults.inl"),
            ("ToneMappingConfig", "white_point", 2.5, "RaytracingConfigDefaults.inl"),
        ):
            changed = copy.deepcopy(SCHEMA)
            item = next(t for t in changed["render_types"] if t["name"] == target)
            next(f for f in item["fields"] if f["name"] == member)["default"] = value
            paths = [p for p, text in GEN.generate(changed).items() if text != base[p]]
            self.assertEqual(sorted(Path(p).name for p in paths), sorted([expected, "template.MoerEngine.toml"]))
        changed = copy.deepcopy(SCHEMA)
        changed["groups"][0]["fields"][0]["default"] = 1280
        paths = [p for p, text in GEN.generate(changed).items() if text != base[p]]
        self.assertEqual(sorted(Path(p).name for p in paths), ["GlobalConfigDefaults.inl", "template.MoerEngine.toml"])

    def test_new_field_updates_all_four_surfaces(self):
        changed = copy.deepcopy(SCHEMA)
        item = next(t for t in changed["render_types"] if t["name"] == "RasterConfig")
        item["fields"].append({"name": "new_setting", "type": "bool", "default": False})
        outputs = GEN.generate(changed)
        affected = [Path(p).name for p, text in outputs.items() if "new_setting" in text]
        self.assertEqual(sorted(affected), sorted(["RasterConfig.generated.h", "RasterConfigDefaults.inl",
                                                  "RasterConfigLoad.inl", "template.MoerEngine.toml"]))

    def test_rejects_invalid_schema_before_writing(self):
        for mutation in ("duplicate", "keyword", "nested_collision", "range", "nonfinite", "enum", "size", "cycle"):
            changed = copy.deepcopy(SCHEMA)
            first = changed["groups"][0]["fields"][0]
            raster = next(t for t in changed["render_types"] if t["name"] == "RasterConfig")
            if mutation == "duplicate":
                changed["groups"][0]["fields"].append(copy.deepcopy(first))
            elif mutation == "keyword":
                first["name"] = "operator"
            elif mutation == "nested_collision":
                changed["groups"][1]["groups"][0]["cpp_type"] = "threading"
            elif mutation == "range":
                first["default"] = -1
            elif mutation == "nonfinite":
                next(f for f in raster["fields"] if f["name"] == "ssao_intensity")["default"] = float("inf")
            elif mutation == "enum":
                next(f for f in raster["fields"] if f["name"] == "aa_mode")["default"] = "Invalid"
            elif mutation == "size":
                next(f for f in raster["fields"] if f["name"] == "probe_gi_volumes")["count"] = 5
            else:
                changed["render_types"][0]["fields"][0]["type"] = "RasterConfig"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                GEN.generate(changed)

    def test_utf8_and_control_escapes(self):
        text = 'a"\\\n\t\x01\u4e2d\U0001f680'
        GEN.validate_value("string", text, "test", {})
        self.assertEqual(json.loads(GEN.scalar("string", text)), text)
        self.assertIn("\\344\\270\\255", GEN.scalar("string", text, True))
        self.assertIn("\\360\\237\\232\\200", GEN.scalar("string", text, True))

    def test_runtime_state_is_not_serialized(self):
        outputs = GEN.generate(SCHEMA)
        template = outputs["template.MoerEngine.toml"]
        for name in ("should_frame_gaussians", "selected_frame_buffer_index", "ai_trt_visualize_buffer_idx", "b_export"):
            self.assertNotIn(name + " =", template)
            self.assertFalse(any(name + ");" in text for path, text in outputs.items() if path.endswith("Load.inl")))

    def test_cli_idempotence_and_check_mode(self):
        with tempfile.TemporaryDirectory(prefix="moer-config-codegen-") as directory:
            root = Path(directory)
            command = [sys.executable, str(HERE / "generate.py"), "--root", str(root)]
            def run(*arguments):
                return subprocess.run(command + list(arguments), capture_output=True, text=True, timeout=30)
            self.assertEqual(run().returncode, 0)
            outputs = GEN.generate(SCHEMA)
            before = {path: (root / path).stat().st_mtime_ns for path in outputs}
            self.assertEqual(run().returncode, 0)
            self.assertEqual(run("--check").returncode, 0)
            self.assertEqual(before, {path: (root / path).stat().st_mtime_ns for path in outputs})
            template = root / "template.MoerEngine.toml"
            template.write_text("stale\n", encoding="utf-8")
            self.assertNotEqual(run("--check").returncode, 0)
            self.assertEqual(template.read_text(encoding="utf-8"), "stale\n")
            template.unlink()
            self.assertNotEqual(run("--check").returncode, 0)
            self.assertEqual(run().returncode, 0)
            self.assertEqual(run("--check").returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
