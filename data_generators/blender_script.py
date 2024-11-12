import bpy
import sys


bpy.ops.wm.read_homefile()
bpy.ops.import_mesh.stl(filepath = sys.argv[4])

pi = 3.14159265359

# camera
bpy.ops.object.camera_add(location = (-200.0, -400.0, 4.0), rotation = (pi/2, 0.0, pi/180 * 330))
bpy.data.cameras["Camera"].clip_end = 5000

# lamp
bpy.ops.object.lamp_add(type = 'POINT',location = (-23.0, -30, 25.0), rotation = (pi/2, 0.0, -pi/2))
bpy.data.lamps[0].energy = 30      

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
      bpy.data.scenes[sceneKey].render.filepath = r'C:\Users\PATH\TO\FOLDER' + str(c)
      
      # resolution of rendered image
      bpy.data.scenes[sceneKey].render.resolution_x = 1920
      bpy.data.scenes[sceneKey].render.resolution_y = 1080
      bpy.data.scenes[sceneKey].render.resolution_percentage = 100

      # crop image (since original camera images are cropped too)
      bpy.data.scenes[sceneKey].render.use_border = True
      bpy.data.scenes[sceneKey].render.border_min_x = 690/1920        
      bpy.data.scenes[sceneKey].render.border_max_x = 1001/1920
      bpy.data.scenes[sceneKey].render.border_min_y = 510/1080
      bpy.data.scenes[sceneKey].render.border_max_y = 650/1080
      bpy.data.scenes[sceneKey].render.use_crop_to_border = True
      bpy.data.scenes[sceneKey].render.image_settings.color_mode = 'BW'

      # Render Scene and store the scene
      bpy.ops.render.render( write_still=True, use_viewport=True )
      c = c + 1
      
print(sys.argv)
