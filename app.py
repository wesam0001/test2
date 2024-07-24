from flask import Flask, render_template, Response
from cvzone.PoseModule import PoseDetector
import cv2

app = Flask(__name__)

# Initialize the webcam to camera
cap = cv2.VideoCapture(0)  # Replace with your video path or 0 for webcam

# Initialize the PoseDetector class with the given parameters
detector = PoseDetector(staticMode=False,
                        modelComplexity=1,
                        smoothLandmarks=True,
                        enableSegmentation=True,
                        smoothSegmentation=True,
                        detectionCon=0.7,
                        trackCon=0.4)

SP = "Starting Position"  # Text for Starting Position (SP)
BP = "Correct Back Position!!"  # Text for Back Position (BP)
KP = "Correct Knee Position!!"  # Text for Knee Position (KP)
IBP = "Incorrect Back Position!!"  # Text for Incorrect Back Position (IBP)
IKP = "Incorrect Knee Position!!"  # Text for Incorrect Knee Position (IKP)

font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
fontScale = 1  # Font scale (size of the font)
RP = (0, 255, 0)  # (BGR) Color of the Right Pose (RP)
WP = (0, 0, 255)  # (BGR) Color of the Wrong Pose (WP)
thickness = 3  # Thickness of the lines used to draw the text


def generate_frames():
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        if not success:
            break

        # Find human pose in the frame
        img = detector.findPose(img)

        # Find the landmarks, bounding box, and center of the body in the frame
        lmList, bboxInfo = detector.findPosition(img, draw=False, bboxWithHands=False)

        # Check if any body landmarks are detected
        if lmList:
            # Calculate the Right Back angle and draw it on the image
            RBack_angle, img = detector.findAngle(lmList[11][0:2],
                                                  lmList[23][0:2],
                                                  lmList[25][0:2],
                                                  img=img,
                                                  color=(0, 0, 255),
                                                  scale=1)

            # Calculate the Right Knee angle and draw it on the image
            RKnee_angle, img = detector.findAngle(lmList[23][0:2],
                                                  lmList[25][0:2],
                                                  lmList[27][0:2],
                                                  img=img,
                                                  color=(0, 0, 255),
                                                  scale=1)

            # Checking if the setting position is correct and write a message on the screen
            if ((RBack_angle > 170 and RBack_angle <= 185)):
                cv2.putText(img, SP, (50, 50), font, fontScale, (255, 0, 0), thickness, cv2.LINE_AA)

            # Checking if the Back and Knee position is Incorrect and write a message on the screen
            elif ((RBack_angle > 295 or RBack_angle < 280)):
                cv2.putText(img, IBP, (50, 100), font, fontScale, WP, thickness, cv2.LINE_AA)
                if ((RKnee_angle > 80 or RKnee_angle < 70)):
                    cv2.putText(img, IKP, (50, 150), font, fontScale, WP, thickness, cv2.LINE_AA)

            # Checking if the Back and Knee position is correct and write a message on the screen
            elif ((RBack_angle > 280 and RBack_angle <= 295)):
                cv2.putText(img, BP, (50, 100), font, fontScale, RP, thickness, cv2.LINE_AA)
                if ((RKnee_angle < 80 and RKnee_angle >= 70)):
                    cv2.putText(img, KP, (50, 150), font, fontScale, RP, thickness, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
