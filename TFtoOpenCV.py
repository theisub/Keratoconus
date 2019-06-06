import cv2

def anotherOne(filename,tempfile):
        # Load a model imported from Tensorflow
        tensorflowNet = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'model.pbtxt')

        # Input image
        img = cv2.imread(filename)
        rows, cols, channels = img.shape
        
        # Use the given image as input, which needs to be blob(s).
        tensorflowNet.setInput(cv2.dnn.blobFromImage(img, size=(347,679), swapRB=True, crop=False))
        
        # Runs a forward pass to compute the net output
        networkOutput = tensorflowNet.forward()
        
        # Loop on the outputs
        for detection in networkOutput[0,0]:
                score = float(detection[2])
                if score > 0.9:
                        left = detection[3] * cols
                        top = detection[4] * rows
                        right = detection[5] * cols
                        bottom = detection[6] * rows
                        
                        #draw a red rectangle around detected objects
                        #cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 111, 255), thickness=1)
                
        # Show the image with a rectagle surrounding the detected objects 
        cv2.imshow('Image', img)
        roi = img[int(top):int(bottom), int(left):int(right)]
        cv2.imwrite(tempfile, roi)
        cv2.waitKey()
        cv2.destroyAllWindows()
        return filename,tempfile
if __name__ == "__main__":
        anotherOne('test.jpg','ehlol.jpg')