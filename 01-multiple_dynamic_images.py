import cv2
import numpy as np
import csv
import numpy as np
from pathlib import Path
from dynamicimage import get_dynamic_image

WINDOW_LENGTH = 15
STRIDE = 10

def main():
    data_path = Path(r'Dataset/Celebdf_resized_resolution',)
    out_path = Path(r'Dataset/Multiple_dynamic_images_stride_10_frames_15')
    out_path.mkdir()
    data_path = Path(data_path)
    #print(f'Load data[{data_path.resolve()}]...')
    categories = list(data_path.glob('*/'))
    #print(categories)
    #word = str(categories[0])
    #word1=word[-10:]
    #print(word.split('*/'))
    za=0
    for subfolder in categories:
        word = str(categories[za])
        word = word[-10:]
        za +=1
        out_category_subfolder = out_path / subfolder.stem
        out_category_subfolder.mkdir()
        #filename = out_category_subfolder
        frames = np.array([cv2.imread(str(x)) for x in subfolder.glob('*.jpg')])
        for i in range(0, len(frames) - WINDOW_LENGTH, STRIDE):
            chunk = frames[i:i + WINDOW_LENGTH]
            assert len(chunk) == WINDOW_LENGTH

            dynamic_image = get_dynamic_image(chunk)
      
            cv2.imwrite(str(out_category_subfolder / (word+str(i).zfill(3)+'.jpg')), dynamic_image)

if __name__ == '__main__':
    main()
