import numpy as np
import tensorflow as tf
import cv2 as cv
import time


def GetROI(filename,tempfile):
    # Read the graph.
    overall = 0
    start = time.time()
    with tf.gfile.FastGFile('frozen_inference_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    finish = time.time()-start
    overall +=finish
    print("GRAPH" + str(finish))
    with tf.Session() as sess:
        # Restore session
        start = time.time()
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        finish = time.time()-start
        print("Importing" + str(finish))
        overall +=finish

        # Read and preprocess an image.
        img = cv.imread(filename)
        rows = img.shape[0]
        cols = img.shape[1]
        inp = img
        #inp = cv.resize(img, (500, 400))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        # Run the model
        start = time.time()
        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                        sess.graph.get_tensor_by_name('detection_scores:0'),
                        sess.graph.get_tensor_by_name('detection_boxes:0'),
                        sess.graph.get_tensor_by_name('detection_classes:0')],
                    feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
        finish = time.time() - start
        overall +=finish
        print("Running" + str(finish))
        # Visualize detected bounding boxes.
        num_detections = int(out[0][0])
        for i in range(num_detections):
            classId = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            if score > 0.95:
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                #cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)

    roi = img[int(y):int(bottom), int(x):int(right)]
    cv.imwrite(tempfile, roi)
    cv.imshow('TensorFlow MobileNet-SSD', roi)
    cv.waitKey()
    return filename, tempfile

if __name__ == "__main__":
    GetROI('test.jpg','temproi.jpg')