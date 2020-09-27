import cv2

cam = cv2.VideoCapture(0)

cv2.namedWindow("Capture image")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("Can't read frame!")
        break
    cv2.imshow("Capture image", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Closing p rograme!")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame[0:200, 0:200])
        print("{} was written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
