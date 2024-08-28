"""HANDTRACKING 1ST PROJECT
 BY :   S6R
 IG: https://www.instagram.com/exoocian/
 """




import cv2
import mediapipe as mp
import numpy as np
import os


class handDetector:
    def __init__(self, mode=False, maxHands=2, minDetectionConf=0.5, minTrackingConf=0.5):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=mode,
            max_num_hands=maxHands,
            min_detection_confidence=minDetectionConf,
            min_tracking_confidence=minTrackingConf
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def isHandClosed(self):
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                landmarks = handLms.landmark
                thumb_tip = landmarks[4]  # Point du bout du pouce
                index_tip = landmarks[8]  # Point du bout de l'index

                # Calcul de la distance entre le bout du pouce et celui de l'index
                distance = np.sqrt((thumb_tip.x - index_tip.x) ** 2 +
                                   (thumb_tip.y - index_tip.y) ** 2 +
                                   (thumb_tip.z - index_tip.z) ** 2)

                # Si la distance est inférieure à un certain seuil, la main est considérée comme fermée
                if distance < 0.05:
                    return True
        return False


def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findHands(img)

        # Vérifie si la main est fermée
        if detector.isHandClosed():
            print("Main fermée, mise en veille de l'ordinateur")
            cap.release()
            cv2.destroyAllWindows()

            # Commande pour mettre en veille
            os.system('rundll32.exe powrprof.dll,SetSuspendState 0,1,0')  # pour Windows
            # os.system('pmset sleepnow')  # pour macOS
            # os.system('systemctl suspend')  # pour Linux

            break

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()