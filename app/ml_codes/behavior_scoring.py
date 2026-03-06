import os
import json
import pickle
import cv2
import numpy as np
from deepface import DeepFace


# ================================
# VIDEO INPUT PATH (EDIT HERE)
# ================================

VIDEO_PATH = r"C:\Users\AdityaRaj\OneDrive\Desktop\InternProject\Interview Room\Scoring_Engine_Report_Generation\data\users\input\video\response3.mp4"


# ================================
# PATH CONFIG
# ================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "ml_models", "cv_model.pkl")

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "users",
    "output",
    "cv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================
# LOAD MODEL CONFIG
# ================================

with open(MODEL_PATH, "rb") as f:
    config = pickle.load(f)

face_detector = cv2.CascadeClassifier(config["face_cascade_path"])
eye_detector = cv2.CascadeClassifier(config["eye_cascade_path"])

weights = config["attention_weights"]
behavior_cfg = config["behavior_config"]


# ================================
# FILE INDEX GENERATOR
# ================================

def get_next_index():

    files = os.listdir(OUTPUT_DIR)

    nums = []

    for f in files:

        if f.startswith("cv_AA") and f.endswith(".json"):

            num = int(f[5:9])
            nums.append(num)

    next_num = max(nums) + 1 if nums else 1

    return f"cv_AA{next_num:04d}.json"


# ================================
# VIDEO ANALYSIS
# ================================

def analyze_video(video_file):

    cap = cv2.VideoCapture(video_file)

    total = 0
    face_frames = 0
    stable = 0
    centered = 0
    eye_contact = 0
    nervous = 0

    prev_center = None

    emotions_sum = {
        "angry":0,
        "disgust":0,
        "fear":0,
        "happy":0,
        "sad":0,
        "surprise":0,
        "neutral":0
    }

    emo_frames = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        total += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray,1.3,5)

        if len(faces) > 0:

            face_frames += 1

            x,y,w,h = faces[0]

            center = (x+w//2,y+h//2)

            frame_center = (
                frame.shape[1]//2,
                frame.shape[0]//2
            )

            # stability
            if prev_center is not None:

                move = np.linalg.norm(
                    np.array(center)-np.array(prev_center)
                )

                if move < behavior_cfg["nervous_movement_threshold"]:
                    stable += 1
                else:
                    nervous += 1

            prev_center = center

            # camera engagement

            dist = np.linalg.norm(
                np.array(center)-np.array(frame_center)
            )

            if dist < 120:
                centered += 1

            # eye contact

            roi = gray[y:y+h,x:x+w]

            eyes = eye_detector.detectMultiScale(roi)

            if len(eyes) >= 2 and dist < 120:
                eye_contact += 1


            # ========================
            # EMOTION DETECTION
            # ========================

            if total % 5 == 0:

                try:

                    face_img = frame[y:y+h,x:x+w]

                    result = DeepFace.analyze(
                        face_img,
                        actions=["emotion"],
                        enforce_detection=False,
                        detector_backend="opencv"
                    )

                    if isinstance(result,list):
                        emotion = result[0]["emotion"]
                    else:
                        emotion = result["emotion"]

                    for k in emotions_sum:
                        emotions_sum[k] += float(emotion[k])

                    emo_frames += 1

                except:
                    pass

    cap.release()


    # ================================
    # ATTENTION SCORE
    # ================================

    fp = face_frames / max(total,1)
    st = stable / max(face_frames,1)
    ce = centered / max(face_frames,1)
    ec = eye_contact / max(face_frames,1)

    movement_control = (st + ce) / 2

    attention = (
        weights["face_present"]*fp +
        weights["stability"]*st +
        weights["camera_engagement"]*ce +
        weights["eye_contact"]*ec +
        weights["movement_control"]*movement_control
    )

    attention_score = {

        "face_present":round(float(fp),2),
        "stability":round(float(st),2),
        "camera_engagement":round(float(ce),2),
        "eye_contact":round(float(ec),2),
        "movement_control":round(float(movement_control),2),
        "overall_attention_score":round(float(attention),2)
    }


    # ================================
    # EMOTION SCORE
    # ================================

    behavior = {}

    for k in emotions_sum:

        if emo_frames > 0:
            behavior[k] = round(emotions_sum[k]/emo_frames,2)
        else:
            behavior[k] = 0


    nervous_ratio = nervous / max(face_frames,1)

    behavior["nervous_movement"] = round(float(nervous_ratio),2)


    positive = behavior["happy"] + behavior["neutral"]

    negative = (
        behavior["angry"] +
        behavior["fear"] +
        behavior["sad"] +
        nervous_ratio*100
    )

    overall_behavior = positive/(positive+negative+1e-6)

    behavior["overall_behavior_score"] = round(float(overall_behavior),2)


    # ================================
    # REMARK
    # ================================

    if overall_behavior > 0.7:
        remark = "Candidate appeared calm and confident."

    elif overall_behavior > 0.5:
        remark = "Candidate showed moderate confidence."

    else:
        remark = "Candidate appeared nervous or distracted."


    return {

        "attention_score":attention_score,
        "behavior_score":behavior,
        "remark":remark
    }


# ================================
# MAIN PROCESS
# ================================

def process_video():

    if not os.path.exists(VIDEO_PATH):

        output = {
            "error":"video is not found"
        }

    else:

        output = analyze_video(VIDEO_PATH)


    filename = get_next_index()

    output_path = os.path.join(OUTPUT_DIR,filename)

    with open(output_path,"w") as f:
        json.dump(output,f,indent=4)

    print("✅ Output Generated:")
    print(output_path)


if __name__ == "__main__":
    process_video()
