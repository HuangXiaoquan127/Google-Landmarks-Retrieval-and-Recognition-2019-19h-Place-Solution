# add env variance DISPLAY=:0 test

import numpy as np
import cv2 as cv
img1_path = '/media/iap205/Data4T/Export_temp/landmarks_view/202510/2c362cf42c944b61.jpg'
img = cv.imread(img1_path,0)
cv.imshow('image',img)
k = cv.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv.imwrite('messigray.png',img)
    cv.destroyAllWindows()