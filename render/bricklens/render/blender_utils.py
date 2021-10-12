import os
import shutil
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
    positionOnGround=False,
    addEnvironment=False,
)
bpy.ops.wm.save_as_mainfile(filepath="{blender_file}")
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as pyout:
        pyout.write(pythonTemplate)
        pyout.close()
        subprocess.run(
            ["blender", "-b", "--python", pyout.name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    os.remove(pyout.name)


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
                # Uncomment the following to use CUDA.
                #"--",
                #"--cycles-device",
                #"CUDA",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        rendered = os.path.join(tmpdir, "img_0001.png")
        if not os.path.exists(rendered):
            print(f"WARNING! Cannot find rendered image {rendered}")
            print(os.listdir(tmpdir))
            raise RuntimeError(f"Rendered image {rendered} is missing")
        else:
            print(f"MDW: RENAMING {rendered} to {output_file}")
            os.rename(rendered, output_file)


def render_ldr(
    ldr_file: str, output_file: str, template_file: str, ldraw_library_path: str
):
    """Render the given LDR file to the given output file."""

    with tempfile.NamedTemporaryFile(suffix=".blend", delete=False) as blender_file:
        ldr_to_blender(ldr_file, blender_file.name, template_file, ldraw_library_path)
        blender_render(blender_file.name, output_file)
    os.remove(blender_file.name)
