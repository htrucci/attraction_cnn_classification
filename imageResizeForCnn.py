from PIL import Image, ImageOps
import os,sys
import glob

attractionList = ['tower eiffel', 'statue of liberty', 'niagara falls', 'colosseum', 'pyramid']
attractionFolderList = ['eiffel', 'liberty', 'niagara', 'colosseum', 'pyramid']
for attractionFolder in attractionFolderList:
    image_dir = "./data_orign/"+attractionFolder+"/"
    target_resize_dir = "./data/"+attractionFolder+"/"
    target_rotate_dir = "./data_rotate/"+attractionFolder+"/"
    if not os.path.isdir(target_resize_dir):
        os.makedirs(target_resize_dir)
    if not os.path.isdir(target_rotate_dir):
        os.makedirs(target_rotate_dir)
    files = glob.glob(image_dir+"*.*")
    print(len(files))
    count = 1;
    size = (224, 224)
    for file in files:
        im = Image.open(file)
        im = im.convert('RGB')
        print("i: ", count, im.format, im.size, im.mode, file.split("/")[-1])
        count+=1
        im = ImageOps.fit(im, size, Image.ANTIALIAS, 0, (0.5, 0.5))
        im.save(target_resize_dir+file.split("/")[-1].split(".")[0]+".jpg", quality=100)
        im.rotate(90).save(target_rotate_dir+"resize_"+file.split("/")[-1].split(".")[0]+".jpg", quality=100)
