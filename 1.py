# python-3.9.0

import sys
import face_recognition
import cv2
import os
import glob
import mediapipe as mp
import numpy as np
import datetime


# Mediapipe to find the bounding box of face due to limitation from face_recognition to detect faces wearing masks
def mediapipe_face(image):
    mp_face_detection = mp.solutions.face_detection
    height, width, channels = image.shape
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.6) as face_detection:
        # Ensure image is in RGB
        results = face_detection.process(image)
        if not results.detections:
            return None
        locations = []
        for data_point in results.detections:
            xmin = int(data_point.location_data.relative_bounding_box.xmin * width)
            ymin = int(data_point.location_data.relative_bounding_box.ymin * height)
            xmax = int((data_point.location_data.relative_bounding_box.xmin +
                        data_point.location_data.relative_bounding_box.width) * width)
            ymax = int((data_point.location_data.relative_bounding_box.ymin +
                        data_point.location_data.relative_bounding_box.height) * height)
            location = (ymin, xmax, ymax, xmin)
            locations.append(location)

        # returns the coordinates for bounding box
        return locations


# Opens webcam and capture the last image before it is terminated
def capture_image():
    image = []
    cap = cv2.VideoCapture(0)
    if not (cap.isOpened()):
        sys.exit("Error: Camera Not Found.")
    else:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading move video, use 'break' instead of 'continue'.
                continue
            image = cv2.flip(image, 1)
            cv2.imshow("Camera", image)

            if cv2.waitKey(1) == ord(" "):
                break

    # returns last image before termination and transform back from BGR to RGB
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Function to register faces with webcam
def register_encoding_with_webcam():
    current_path = os.getcwd()
    # Locate "users" directory or create one if not found
    users_path = os.path.join(current_path, "Users")
    try:
        os.listdir(users_path)
    except FileNotFoundError:
        os.mkdir(users_path)

    # Loop to add multiple users at once
    while True:
        username = input("Please enter your name: ")
        username_path = os.path.join(users_path, username)

        # Locate or create username_path
        try:
            os.listdir(username_path)
        except FileNotFoundError:
            os.mkdir(username_path)

        # Find all paths to files ending with .npy in .\Users\<Username>
        user_encodings = glob.glob(os.path.join(username_path, "*.npy"))
        number_of_encodings = len(user_encodings)

        print("You currently have {} encodings saved.".format(number_of_encodings))
        add_encoding_response = input("Would you like to add a new encoding? (y/n) ")
        if add_encoding_response == "y":
            while True:
                print("Please position yourself in front of the camera and press <spacebar> to capture the image. ")
                image = capture_image()
                location = mediapipe_face(image)
                result = face_recognition.face_encodings(image, known_face_locations=location)

                # If len(result) > 1 means there is more than person in frame
                # len == 0 means no face detected
                # Loop through so that user can have multiple tries
                while len(result) > 1 or len(result) == 0 or not location:
                    if len(result) > 1:
                        print("Please ensure that there is only one person is in the frame. ")
                    else:
                        print("Please ensure that there is at least a person in the frame. ")
                    image = capture_image()
                    location = mediapipe_face(image)
                    result = face_recognition.face_encodings(image, location)

                # Saves the encoding to .\Users\<Username>\encoding{1/2/3/4/5}.npy
                np.save("{}\encoding{}.npy".format(username_path, number_of_encodings+1), result)
                print("Sucessfully registered.")

                # Loop request
                repeat = input("Would you like to add another new encoding? (y/n) ")
                if repeat == "n":
                    break

            # Loop request
            repeat2 = input("Would you like to continue to add encoding for another user? (y/n) ")
            if repeat2 == "n":
                break


# Same as register_encoding_with_webcam() but uses pictures
def register_encoding_with_pictures():
    current_path = os.getcwd()
    # Locate "users" directory or create one if not found
    users_path = os.path.join(current_path, "Users")
    try:
        os.listdir(users_path)
    except FileNotFoundError:
        os.mkdir(users_path)

    while True:
        username = input("Please enter your name: ")
        username_path = os.path.join(users_path, username)

        # Locate or create "username_path"
        try:
            os.listdir(username_path)
        except FileNotFoundError:
            os.mkdir(username_path)

        user_encodings = glob.glob(os.path.join(username_path, "*.npy"))
        number_of_encodings = len(user_encodings)

        print("You currently have {} encodings saved.".format(number_of_encodings))
        add_encoding_response = input("Would you like to add a new encoding? (y/n) ")
        if add_encoding_response == "y":
            while True:
                image = input("Please enter absolute path to image: ")
                image = cv2.imread(image)
                location = mediapipe_face(image)
                result = face_recognition.face_encodings(image, known_face_locations=location)
                while len(result) > 1 or len(result) == 0 or not location:
                    if len(result) > 1:
                        print("Please ensure that there is only one person in the picture. ")
                    else:
                        print("Please ensure that there is at least a person in the picture. ")
                    image = input("Please enter absolute path to image: ")
                    image = cv2.imread(image)
                    location = mediapipe_face(image)
                    result = face_recognition.face_encodings(image, location)

                np.save("{}\encoding{}.npy".format(username_path, number_of_encodings + 1), result)
                print("Sucessfully registered.")
                repeat = input("Would you like to add another new encoding? (y/n) ")
                if repeat == "n":
                    break
            repeat2 = input("Would you like to continue to add encoding for another user? (y/n) ")
            if repeat2 == "n":
                break


# Compiles all of the encoding and their respective names into memory
# If we know who will be appearing that day, we can only extract their encodings into memory
# Might cause memory issue if too many users registered
def compile_all_encoding():
    users_path = r".\Users"
    list_of_users = os.listdir(users_path)
    names = []
    encodings = []
    for i, user in enumerate(list_of_users):
        user_path = os.path.join(users_path, user)
        list_of_encodings = glob.glob(os.path.join(user_path, "*.npy"))
        for o, path_to_encoding in enumerate(list_of_encodings):
            encoding = np.load(path_to_encoding)
            if i == 0 and o == 0:
                encodings = encoding
                names.append(user)
            else:
                encodings = np.append(encodings, encoding, axis=0)
                names.append(user)
    return names, encodings


# Face recognition with webcam
def live_cam_recognition():
    users, encodings = compile_all_encoding()
    timer = []
    cap = cv2.VideoCapture(0)
    fps_counter = ""
    color = (0, 0, 255)
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    if not (cap.isOpened()):
        sys.exit("Error: Camera Not Found.")
    else:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading move video, use 'break' instead of 'continue'.
                continue
            image = cv2.flip(image, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channels = image.shape

            # Call mediapiper_face() to extract bounding box of face
            locations = mediapipe_face(image)

            # If there is someone in frame, then proceed
            if locations:

                # For multiple people
                for i, location in enumerate(locations):
                    encoding = face_recognition.face_encodings(image, [location])
                    matches = face_recognition.compare_faces(encodings, encoding)
                    names = []

                    # List out all of the matched name
                    for o, p in enumerate(matches):
                        if p:
                            names.append(users[o])

                    # If not one match, then marked as unknown
                    if not names:
                        names = ["Unknown"]

                    # Print out all matched name
                    print(names)

                    # Aesthetic (bounding box)
                    start_point = [location[3], location[2]]
                    end_point = [location[1], location[0]]
                    image = cv2.rectangle(image, start_point, end_point, color, thickness)
                    image = cv2.putText(image, str(names[0]), start_point + np.array([0, 20]), font, fontScale, color,
                                        thickness, cv2.LINE_AA)

            # Aesthetic (fps counter)
            now = datetime.datetime.now().strftime("%H:%M:%S")
            timer.append(now)
            if datetime.datetime.strptime(timer[-1], "%H:%M:%S") - datetime.datetime.strptime(timer[0],"%H:%M:%S")\
                    == datetime.timedelta(seconds=1):
                fps = len(timer)
                fps_counter = "FPS: {}".format(fps)
                timer = []

            cv2.putText(image, fps_counter, [width-75, 0+25], font, fontScale, color, thickness, cv2.LINE_AA)

            # Shows the frame
            cv2.imshow("Camera", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # The loop will break after pressing <spacebar> or can change to any key by changing to ord("<key>")
            if cv2.waitKey(1) == ord(" "):
                break


if __name__ == "__main__":
    #register_encoding_with_webcam()
    #register_encoding_with_pictures()
    #live_cam_recognition()
