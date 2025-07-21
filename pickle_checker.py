import pickle, pathlib, pprint, itertools, textwrap
kp = pickle.load(open(
    '/home/nirmal/SlowFast/GradCAMs/data/phoenix-2014-multisigner/keypoints/keypoints.pkl','rb'))
print(type(kp), len(kp))
# show first element
if isinstance(kp, list):
    pprint.pprint(kp[0].keys())
else:
    print(next(iter(kp.keys()))[:150])