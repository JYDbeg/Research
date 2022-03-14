from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
types = ["jpg","jpeg"]

def test():
    for t in types:
        for index,file in enumerate(glob.glob("netgazou/*.{}".format(t))):
            gray = Image.open(file).convert("L").convert("RGB")
            gray.save(file, quality=95)


if __name__ == "__main__":
    test()