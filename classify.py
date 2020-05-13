import cv2
from PIL import Image
import PIL
import numpy
import numpy as np
def init_classify():
    global net, classes
    rows = open('Model/synset_words.txt').read().strip().split("\n")
    classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
    weight = 'Model/bvlc_googlenet.caffemodel'
    arch = 'Model/bvlc_googlenet.prototxt'
    net = cv2.dnn.readNetFromCaffe(arch, weight)


def predict(image1):

    init_classify()
	
    pil_image = PIL.Image.open(image1).convert('RGB')
    open_cv_image = numpy.array(pil_image) 
    # Convert RGB to BGR 
    image = open_cv_image[:, :, ::-1].copy()
 	
    #image = cv2.imread(open_cv_image)

    # Make prediction
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (224, 224),
                                                (104, 117, 123)))
    net.setInput(blob)
    prediction = net.forward()
    idxs = np.argsort(prediction[0])[::-1][0]
    label = classes[idxs]
    confidence = round(prediction[0][idxs] * 100, 2)
    #comb = label + "," + str(confidence)
    return label,confidence