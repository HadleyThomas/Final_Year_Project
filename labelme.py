
import os, sys, shutil
path_labelme="/Users/Hadley 1/Documents/MEng Project/python/labelme"
dirs = os.listdir(path_labelme)
i = 0
for item in dirs:
   if item.endswith(".json"):
      print(item)
      if os.path.isfile(path_labelme+"/"+item):
         print("NOW")
         file_name = "sat"+str(i)
         my_dest ="/Users/Hadley\ 1/Documents/MEng\ Project/python/renders/" + file_name
         os.system("mkdir "+my_dest)
         os.system("labelme_json_to_dataset "+"/Users/Hadley\ 1/Documents/MEng\ Project/python/labelme/"+item+" -o "+my_dest)
         i+=1
         

path_renders ="/Users/Hadley 1/Documents/MEng Project/python/renders"
directory = os.listdir(path_renders)
i=0
print(directory)
for folder in directory:
	if folder.startswith("sat"):
		print(folder)
		os.chdir(path_renders +"/"+ folder)
		image_name = "image_" + str(i) +".png"
		label_name = "mask_" + str(i) +".png"
		os.rename("img.png", image_name)
		os.rename("label.png", label_name)
		i+=1

i = 0
image_folder = "/Users/Hadley 1/Documents/MEng Project/python/images"
mask_folder = "/Users/Hadley 1/Documents/MEng Project/python/masks"
for folder in directory:
	if folder.startswith("sat"):
		image = "image_"+str(i)+".png"
		mask = "mask_"+str(i)+".png"
		shutil.move(path_renders+"/"+folder+"/"+image, image_folder)
		shutil.move(path_renders+"/"+folder+"/"+mask, mask_folder)
		i+=1







