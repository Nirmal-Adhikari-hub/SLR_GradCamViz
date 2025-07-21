import pickle
import json
import numpy as np   # only needed if you want to detect numpy arrays

def summarize(obj, depth=0, max_depth=3):
    """
    Return a nested Python structure describing types/sizes.
    At max_depth we stop descending and just report the type.
    """
    if depth >= max_depth:
        return type(obj).__name__

    # dict → map keys to summaries
    if isinstance(obj, dict):
        return {k: summarize(v, depth+1, max_depth) for k, v in obj.items()}

    # list/tuple → show the first element
    if isinstance(obj, (list, tuple)):
        if not obj:
            return f"{type(obj).__name__} (empty)"
        return [summarize(obj[0], depth+1, max_depth)]

    # numpy arrays → show shape
    if isinstance(obj, np.ndarray):
        return f"ndarray{obj.shape}"

    # anything else → just show the type
    return type(obj).__name__


if __name__ == '__main__':
    path = '/home/nirmal/SlowFast/GradCAMs/data/phoenix-2014-multisigner/keypoints/keypoints.pkl'
    with open(path, 'rb') as f:
        kp = pickle.load(f)

    summary = summarize(kp, max_depth=4)
    print(json.dumps(summary, indent=2))