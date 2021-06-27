import bpy

bpy.ops.wm.open_mainfile(filepath="emptyscene.blend")
bpy.ops.import_scene.importldraw(
    filepath="mess.ldr",
    ldrawPath="/home/mdw/src/downloads/ldraw",
    importCameras=False,
    positionCamera=False,
    addEnvironment=False,
)
bpy.ops.wm.save_as_mainfile(filepath="testrender.blend")
