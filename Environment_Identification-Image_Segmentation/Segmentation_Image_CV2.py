# Import libraries
import cv2
import numpy as np

# Load the COCO 80 class model
rcnn = cv2.dnn.readNetFromTensorflow('DNN/frozen_inference_graph_coco.pb',
                                     'DNN/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt')

# Read the image
img = cv2.imread('img2.jpeg')
alto, ancho, _ =  img.shape
#print(alto, ancho)

# Generate the colors
colores = np.random.randint(0, 255, (80,3))

# Prepare our image
blob = cv2.dnn.blobFromImage(img, swapRB = True) # Swap: BGR -> RGB

# Process the image with the model
rcnn.setInput(blob)

# Extract the Rect and Masks
info, masks = rcnn.forward(["detection_out_final", "detection_masks"])

# Extract the number of detected objects
contObject = info.shape[2]
#print(contObject)

# Iterate over the detected objects
for i in range(contObject):
    # Extract the rectangles from the objects
    inf = info[0,0,i]
    #print(inf)

    # Extract the class of image
    clase = int(inf[1])
    #print(clase)

    # extract score belonging to the class
    puntaje = inf[2]

    # Filter to eliminate small detections
    if puntaje < 0.7:
        continue

    # Rectangle coordinates for object detection
    x = int(inf[3] * ancho)
    y = int(inf[4] * alto)
    x2 = int(inf[5] * ancho)
    y2 = int(inf[6] * alto)

    # Extract the size of the objects
    tamobj = img[y:y2, x:x2]
    tamalto, tamancho, _ = tamobj.shape
    #print(tamalto, tamancho)

    # Extract Mask 
    mask = masks[i, clase]
    mask = cv2.resize(mask, (tamancho, tamalto))

    # Set a threshold
    _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
    mask = np.array(mask, np.uint8)
    #print(mask.shape)

    # Extract coordinates from the mask
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Choose the colors for each class
    color = colores[clase]
    r = int(color[0])
    g = int(color[1])
    b = int(color[2])

    # Iterate the contours
    for cont in contornos:
        # Draw mask
        cv2.fillPoly(tamobj, [cont], (r,g,b))
        cv2.rectangle(img, (x, y), (x2, y2), (r, g, b), 3)

        #print(cont)
        
    # Show the image
    cv2.imshow('TAMANO OBJETO', tamobj)
    # cv2.imshow('MASCARA', mask)
    cv2.waitKey(0)

cv2.imshow('IMAGEN', img)
cv2.waitKey(0)