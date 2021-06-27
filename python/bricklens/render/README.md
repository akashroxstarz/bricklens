

./render.py > test.ldr

./ldraw/tools/ldr2png.py ~/src/github/teamsidney/brickfinder/render/test.ldr --stroke-colour '#000000' foo.png 10240x7680 2000,2000,-2000

povray -ifoo.pov +W12800 +H10240 +fp -o-  > o.png


```
blender -b mess2.blend -f 1 -o //mess2_ -- --cycles-device CUDA
```

https://blender.stackexchange.com/questions/73759/install-addons-in-headless-blender

To install addon via command line:

```
blender -b --python-console
>>> import bpy
>>> bpy.ops.preferences.addon_install(filepath='/home/mdw/src/bricklens/python/bricklens/render/importldraw1.1.11_for_blender_281.zip')
>>> bpy.ops.preferences.addon_enable(module='io_scene_importldraw')
>>> bpy.ops.wm.save_userpref()
>>> import addon_utils
>>> addon_utils.check('io_scene_importldraw')
(True, True)
```

Running LDraw importer via Python script:

```
import bpy

bpy.ops.import_scene.importldraw(filepath="mess.ldr", ldrawPath="/home/mdw/src/downloads/ldraw")

```

```
blender -b --python blender_render.py
blender -b testrender.blend -f 1 -o //mess2_ -- --cycles-device CUDA
```
