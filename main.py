import cv2 as cv
import mediapipe as mp
import time


class facedetection:
    def __init__(self, min_detection_confidence=0.5):
        self.mdc = min_detection_confidence
        # Yüz tespiti için model yükleniyor
        self.mpfaces = mp.solutions.face_detection
        self.faces = self.mpfaces.FaceDetection(self.mdc)

    def findface(self, frame, draw=True):
        RGBimg = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = self.faces.process(RGBimg)

        if result.detections:
            for id, detection in enumerate(result.detections):
                bbox = detection.location_data.relative_bounding_box
                h, w, c = frame.shape
                bboxs = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
                cv.rectangle(frame, bboxs, (0, 0, 255), 3)
                cv.putText(frame, f"{int(detection.score[0] * 100)}", (bboxs[0], bboxs[1] - 20), cv.FONT_HERSHEY_PLAIN,
                           3, (255, 0, 0), 3)
        return frame


def main():
    cap = cv.VideoCapture('vid3.mp4')
    cTime = 0
    pTime = 0
    detector = facedetection()

    # window sizes
    window_width = 720
    window_height = 480

    while True:
        success, frame = cap.read()

        if not success:
            break

        # reshaping the window
        frame = cv.resize(frame, (window_width, window_height))

        frame = detector.findface(frame)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(frame, f"{int(fps)} FPS", (15, 40), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        # reshaping the size
        cv.imshow('image', frame)

        if cv.waitKey(20) & 0xFF == ord('d'):
            break

    cap.release()
    cv.destroyAllWindows()


# end of class
if __name__ == "__main__":
    main()
