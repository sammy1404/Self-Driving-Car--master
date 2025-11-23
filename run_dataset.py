import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


import cv2
import numpy as np
import model
import math

# --- Load Model ---
sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")

# Load steering wheel image
wheel_img = cv2.imread('steering_wheel_image.jpg', 0)
rows, cols = wheel_img.shape

smoothed_angle = 0

# --- Open Video Feed ---
# For webcam use: cv2.VideoCapture(0)
cap = cv2.VideoCapture("Lane Dataset (1).mp4")

if not cap.isOpened():
    print("Error opening video source")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame from BGR to RGB (model expects RGB)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Crop lower half or appropriate region (adjust as needed)
    crop = rgb[-150:]

    # Resize to (66, 200) as NVIDIA model expects
    image = cv2.resize(crop, (200, 66)) / 255.0

    # Predict steering angle
    degrees = model.y.eval(
        feed_dict={model.x: [image], model.keep_prob: 1.0}
    )[0][0] * 180.0 / np.pi

    print("Predicted steering angle:", degrees)

    # --- Smooth steering ---
    if degrees != smoothed_angle:
        smoothed_angle += 0.2 * pow(abs(degrees - smoothed_angle), 2.0 / 3.0) * \
                          (degrees - smoothed_angle) / abs(degrees - smoothed_angle)

    # Rotate steering wheel image
    M = cv2.getRotationMatrix2D((cols/2, rows/2), -smoothed_angle, 1)
    rotated = cv2.warpAffine(wheel_img, M, (cols, rows))

    # Show video + steering wheel
    cv2.imshow("Video Feed", frame)
    cv2.imshow("Steering Wheel", rotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
