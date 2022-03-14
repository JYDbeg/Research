from PIL import Image
import numpy as np
a = Image.open("ha_j/f01/happiness/no90kao/f01_hap_0.jpg")
b =Image.open("ha_v/f01/happiness/sample_video_img_74.jpg")
a = np.array(a)
b = np.array(b)
print(a-b)
from matplotlib import pyplot as plt
plt.imshow(a-b)
plt.show()