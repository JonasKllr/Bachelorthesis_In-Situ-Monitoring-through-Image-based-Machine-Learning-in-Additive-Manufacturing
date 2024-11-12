import bpy, bgl, blf,sys
from bpy import data, ops, props, types, context

pi = 3.14159265359

bpy.ops.wm.read_homefile()

bpy.ops.import_mesh.stl(filepath = sys.argv[4])

# camera 1
# camera position
bpy.ops.object.camera_add(location = (-200.0, -400.0, 4.0), rotation = (pi/2, 0.0, pi/180 * 330))
# lamp type and position
bpy.ops.object.lamp_add(type = 'POINT',location = (-23.0, -30, 25.0), rotation = (pi/2, 0.0, -pi/2))

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

#increase lamp energy
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
      bpy.data.scenes[sceneKey].render.filepath = r'C:\Users\Jonas\Documents\Studium\Bachelor\Bachelorarbeit\Blender\Render\camera_' + str(c)   #Speicherpfad für neues Bild festlegen
      
      #Auflösung von gerendertem Bild
      bpy.data.scenes[sceneKey].render.resolution_x = 1920          
      bpy.data.scenes[sceneKey].render.resolution_y = 1080
      bpy.data.scenes[sceneKey].render.resolution_percentage = 100

      #Bild schneiden
      bpy.data.scenes[sceneKey].render.use_border = True
      bpy.data.scenes[sceneKey].render.border_min_x = 690/1920        
      bpy.data.scenes[sceneKey].render.border_max_x = 1001/1920           # gerenderte Bilder sind zu klein
      bpy.data.scenes[sceneKey].render.border_min_y = 510/1080
      bpy.data.scenes[sceneKey].render.border_max_y = 650/1080
      bpy.data.scenes[sceneKey].render.use_crop_to_border = True
      bpy.data.scenes[sceneKey].render.image_settings.color_mode = 'BW'


      # Render Scene and store the scene
      bpy.ops.render.render( write_still=True, use_viewport=True )
      c = c + 1
      
print(sys.argv)
