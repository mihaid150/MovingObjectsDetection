import cv2
import time
import os


def clean_folder(path):
    for filename in os.listdir(path):
        os.remove(os.path.join(path, filename))


folder_path = "Grayscaled Frames"
clean_folder(folder_path)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

width = 640
height = 480
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)
cap.set(10, 150)

frames = []
start_time = time.time()

while time.time() - start_time < 4:
    success, img = cap.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frames.append(img_gray)

cap.release()

for i in range(len(frames)):
    cv2.imwrite(os.path.join(folder_path, f"frame_{i}.png"), frames[i])

frame_index = 0
while True:
    img = cv2.imread(os.path.join(folder_path, f"frame_{frame_index}.png"))
    cv2.imshow("Frame", img)

    key = cv2.waitKey(0)

    if key == ord(' '):  # Space key is pressed, advance to the next frame
        frame_index += 1
        if frame_index >= len(frames):
            break
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
