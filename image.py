import matplotlib.pyplot as plt
import numpy as np
import sys

image = np.fromfile(sys.argv[1],np.uint8,784,"",(int(sys.argv[2])-1)*784)

image = image.reshape(28,28)
plt.figure(figsize = (3.5,3.5))
plt.xticks([])
plt.yticks([])
plt.imshow(image, cmap="binary")
plt.grid(False)
plt.show()