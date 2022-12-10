import cv2

car_file = "car6.jpg"
car_img = cv2.imread(car_file)
car_gray = cv2.cvtColor(car_img, cv2.COLOR_BGR2GRAY)

video = cv2.VideoCapture("shortvid.mp4")

car_classifier = "car_detector.xml"
pedestrian_classifier = "haarcascade_fullbody.xml"

trained_car_detector = cv2.CascadeClassifier(car_classifier)
trained_pedestrian_detector = cv2.CascadeClassifier(pedestrian_classifier)

while True:
    (successful_Read, frame) = video.read()

    if successful_Read:
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    coordniates_in_video = trained_car_detector.detectMultiScale(grayscaled_frame)
    coords_for_pedes_in_video = trained_pedestrian_detector.detectMultiScale(grayscaled_frame)

    for x, y, w, h in coordniates_in_video:
        cv2.rectangle(frame, (x, y), (x + h, y + w), (0, 255, 0), 2)

    for x, y, w, h in coords_for_pedes_in_video:
        cv2.rectangle(frame, (x, y), (x + h, y + w), (0, 0, 255), 2)

    cv2.imshow("Video Showcase", frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

video.release()

# for an image
# car_coordinates = trained_car_detector.detectMultiScale(car_gray)
# print(car_coordinates)
#
# for x, y, w, h in car_coordinates:
#     cv2.rectangle(car_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
#
# cv2.imshow("Car Detector", car_img)
# cv2.waitKey()

print("Code Completed Successfully")


