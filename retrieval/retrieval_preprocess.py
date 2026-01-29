import cv2
import numpy as np
from scipy.interpolate import interp1d
import mediapipe as mp

mp_holistic = mp.solutions.holistic

N_UPPER_BODY_POSE_LANDMARKS = 25
N_HAND_LANDMARKS = 21

N_TOTAL_LANDMARKS = (
    N_UPPER_BODY_POSE_LANDMARKS
    + N_HAND_LANDMARKS
    + N_HAND_LANDMARKS
)

SEQUENCE_LENGTH = 60


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def extract_keypoints(results):
    """
    Output: (67 * 3,) flattened
    """
    pose_kps = np.zeros((N_UPPER_BODY_POSE_LANDMARKS, 3))
    left_hand_kps = np.zeros((N_HAND_LANDMARKS, 3))
    right_hand_kps = np.zeros((N_HAND_LANDMARKS, 3))

    if results and results.pose_landmarks:
        for i in range(N_UPPER_BODY_POSE_LANDMARKS):
            if i < len(results.pose_landmarks.landmark):
                res = results.pose_landmarks.landmark[i]
                pose_kps[i] = [res.x, res.y, res.z]

    if results and results.left_hand_landmarks:
        left_hand_kps = np.array(
            [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
        )

    if results and results.right_hand_landmarks:
        right_hand_kps = np.array(
            [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
        )

    keypoints = np.concatenate(
        [pose_kps, left_hand_kps, right_hand_kps],
        axis=0
    )

    return keypoints.flatten()


def interpolate_keypoints(keypoints_sequence, target_len=SEQUENCE_LENGTH):
    """
    Input:
        keypoints_sequence: List[np.ndarray (201,)]
    Output:
        np.ndarray (60, 201)
    """
    if len(keypoints_sequence) == 0:
        return None

    original_times = np.linspace(0, 1, len(keypoints_sequence))
    target_times = np.linspace(0, 1, target_len)

    num_features = keypoints_sequence[0].shape[0]
    interpolated_sequence = np.zeros((target_len, num_features))

    for feature_idx in range(num_features):
        feature_values = [frame[feature_idx] for frame in keypoints_sequence]

        interpolator = interp1d(
            original_times,
            feature_values,
            kind='cubic',
            bounds_error=False,
            fill_value="extrapolate"
        )

        interpolated_sequence[:, feature_idx] = interpolator(target_times)

    return interpolated_sequence


def extract_sequence_from_video(
    video_path: str,
    sequence_length: int = SEQUENCE_LENGTH,
    max_sample_frames: int = 100
):
    """
    Pipeline truy hồi:
    video → mediapipe → keypoints → interpolate → sequence

    Return:
        np.ndarray (60, 201)
        hoặc None nếu lỗi
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // max_sample_frames)

    sequence_frames = []

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if frame_idx % step != 0:
                continue

            try:
                _, results = mediapipe_detection(frame, holistic)
                keypoints = extract_keypoints(results)

                if keypoints is not None:
                    sequence_frames.append(keypoints)

            except Exception:
                continue

    cap.release()

    if len(sequence_frames) == 0:
        return None

    sequence = interpolate_keypoints(
        sequence_frames,
        target_len=sequence_length
    )

    return sequence
