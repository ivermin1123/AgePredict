from PIL import Image
import os
from tqdm import tqdm

path1 = "./data/"
dirlist = [ item for item in os.listdir(path1) if os.path.isdir(os.path.join(path1, item))]

print(dirlist)
dirs = []
for item in dirlist:
    pathend = path1 + '//' + item + '//'
    dirs.append(pathend)

for item in tqdm(dirs):
    path = str(item)
    dirs1 = os.listdir(path)
    def resize():
        for item in dirs1:
            if os.path.isfile(path + item):
                im = Image.open(path + item)
                im = im.convert('RGB')
                f, e = os.path.splitext(path + item)
                imResize = im.resize((224, 224), Image.ANTIALIAS)
                imResize.save(f + ' resized.jpg', 'JPEG', quality=90)
                os.remove(path + item)
    resize()
