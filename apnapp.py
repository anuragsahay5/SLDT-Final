from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image
from image_processing import func
import cv2
import numpy as np
import os
import string

# Load the model architecture from the JSON file
with open('model/model-bw.json', 'r') as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)

# Load the weights into the model
loaded_model.load_weights('model/model-bw.h5')

# Compile the loaded model
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Function to preprocess a single image before feeding it to the model
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(sz, sz), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the pixel values
    return img_array


cap = cv2.VideoCapture(0)
interrupt = -1  
i=0
minValue=70 
while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (220-1, 9), (620+1, 419), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[10:410, 220:520]
    
    cv2.imshow("Frame", frame)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray,(5,5),2)
    
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, test_image = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    
    test_image = cv2.resize(test_image, (300,300))
    cv2.imshow("test", test_image)

    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
    if interrupt & 0xFF == ord(' '):
        cv2.imwrite('photu.jpg', roi)
        bw_image = func('photu.jpg')
        cv2.imwrite('photu.jpg', bw_image)
        sz=128
        preprocessed_image = preprocess_image('photu.jpg')

        # Make predictions
        predictions = loaded_model.predict(preprocessed_image)

        # Get the class with the highest probability
        predicted_class = np.argmax(predictions)

        print(f"Predicted Class: {chr(96 + predicted_class)}")
        i+=1
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(roi,  
                    f"{chr(96 + predicted_class)}",  
                    (100, 50),  
                    font, 
                    3,  
                    (0, 0, 0),  
                    2,  
                    cv2.LINE_4)
        cv2.imshow("ROI", roi)
cap.release()
cv2.destroyAllWindows()
