from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image   
import numpy as np
import sys


#---------------------------------------------------#
#image = Image.open(sys.argv[1]).convert('L')
##image = image.crop((500,500,1100,1100))
#image = image.resize((28,28))
#image = np.asarray( image, dtype="uint8" )
#image = 255 - image
#---------------------------------------------------#



model = keras.models.load_model("my_model")



#-----------------------------------------------#
#(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

#train_images = train_images / 255.0
#test_images = test_images / 255.0

#test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
#-----------------------------------------------#



prob_model = keras.Sequential([model,keras.layers.Softmax()])



#-----------------------------------------------#
#image = np.fromfile(sys.argv[1],np.uint8,784,"",(int(sys.argv[2])-1)*784)
#image = image.reshape(28,28)
#-----------------------------------------------#


#image = image / 255.0
#image[image < 0.3] = 0


#---------------------------------------------------#
#plt.figure(figsize = (3.5,3.5))
#plt.xticks([])
#plt.yticks([])
#plt.imshow(image, cmap="binary")
#plt.grid(False)
#plt.show()
#---------------------------------------------------#


image = image.reshape(1,28,28)
prediction = prob_model.predict(image)
print("\n",prediction,"\n\n",prediction.argmax(),"\n")
