import os
import subprocess
import tempfile


def ldr_to_blender(
    ldr_file: str, blender_file: str, template_file: str, ldraw_library_path: str
):
    """Convert an LDR file to a Blender file.

    This function assumes that Blender has been configured with the
    "importldraw" add-on installed.

    Args:
        ldr_file: The path to the LDR file.
        blender_file: The path to the Blender file to generate.
        template_file: The path of a Blender file to use as a template, which should set lighting,
            camera, renmdering settings, etc.
        ldraw_library_path: Directory containing complete LDRAW library.
    """
    pythonTemplate = f"""
import bpy

bpy.ops.wm.open_mainfile(filepath="{template_file}")
bpy.ops.import_scene.importldraw(
    filepath="{ldr_file}",
    ldrawPath="{ldraw_library_path}",
    importCameras=False,
    positionCamera=False,
    addEnvironment=False,
)
bpy.ops.wm.save_as_mainfile(filepath="{blender_file}")
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as pyout:
        pyout.write(pythonTemplate)
        pyout.close()
        print(f"Python file is {pyout.name}")
        subprocess.run(["blender", "-b", "--python", pyout.name])
    os.remove(pyout.name)
    print(f"Generated {blender_file}")


def blender_render(blender_file: str, output_file: str):
    """Render the given Blender file to the given output_file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            [
                "blender",
                "-b",
                blender_file,
                "--render-output",
                f"{tmpdir}/img_",
                "-f",
                "1",
                "--",
                "--cycles-device",
                "CUDA",
            ]
        )
        os.rename(os.path.join(tmpdir, "img_0001.png"), output_file)
    print(f"Saved image to {output_file}")
