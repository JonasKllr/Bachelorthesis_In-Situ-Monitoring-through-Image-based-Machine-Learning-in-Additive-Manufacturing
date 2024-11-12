import bpy, bgl, blf,sys
from bpy import data, ops, props, types, context

pi = 3.14159265359

bpy.ops.wm.read_homefile()

bpy.ops.import_mesh.stl(filepath = sys.argv[4])

# camera 1
bpy.ops.object.camera_add(location = (100.0, 0.0, 15.0), rotation = (pi/2, 0.0, pi/2))
bpy.ops.object.lamp_add(type = 'SUN',location = (-100.0, 0.0, 15.0), rotation = (pi/2, 0.0, -pi/2))

# camera 2
# bpy.ops.object.camera_add(location = (0.0, 100.0, 15.0), rotation = (pi/2, 0.0, pi))
# bpy.ops.object.lamp_add(type = 'SUN',location = (0.0, -100.0, 15.0), rotation = (pi/2, 0.0, 0.0))

# camera 3
# bpy.ops.object.camera_add(location = (0.0, 0.0, 10.0), rotation = (pi/2, 0.0, pi))
# bpy.ops.object.lamp_add(type = 'SUN',location = (0.0, 0.0, 10.0), rotation = (pi/2, 0.0, pi))

# camera 4
# bpy.ops.object.camera_add(location = (0.0, 0.0, 10.0), rotation = (pi/2, 0.0, pi))
# bpy.ops.object.lamp_add(type = 'SUN',location = (0.0, 0.0, 10.0), rotation = (pi/2, 0.0, pi))

#need to increase camera clipping by A LOT
bpy.data.cameras["Camera"].clip_end = 5000
# bpy.data.cameras["Camera.001"].clip_end = 5000
# bpy.data.cameras["Camera.002"].clip_end = 5000

cameraNames=''

print('\nPrint Scenes...')
sceneKey = bpy.data.scenes.keys()[0]
print('Using Scene['  + sceneKey + ']')

# Loop all objects and try to find Cameras
print('Looping Cameras')
c=0
for obj in bpy.data.objects:
    # Find cameras that match cameraNames
    if ( obj.type =='CAMERA') and ( cameraNames == '' or obj.name.find(cameraNames) != -1) :
      print("Rendering scene["+sceneKey+"] with Camera["+obj.name+"]")

      # Set Scenes camera and output filename
      bpy.data.scenes[sceneKey].camera = obj
      #bpy.data.scenes[sceneKey].render.file_format = 'JPEG'
      bpy.data.scenes[sceneKey].render.filepath = './camera_' + str(c)

      # Render Scene and store the scene
      bpy.ops.render.render( write_still=True )
      c = c + 1
      
print(sys.argv)
