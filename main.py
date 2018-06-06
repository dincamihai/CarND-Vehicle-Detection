import os
import math
import uuid
import time
import pickle
import argparse
import matplotlib
matplotlib.use('svg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from functools import partial
from skimage.feature import hog
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.ndimage.measurements import label, find_objects
from scipy.spatial import distance

OUTPUT_FOLDER = './output_images'

HEATMAP_REFRESH_FREQ = 5
MAX_HEATMAP_VALUE = 4
CONFIDENCE_THRESHOLD = 5
CENTROID_CHANGE_RATE = 0.2
MAX_CONFIDENCE = 100
OBJECT_MERGING_DISTANCE = 110
CONFIDENCE_SCALER = 255 // (MAX_CONFIDENCE + 1)
VALUE_SCALER = 255 // (MAX_HEATMAP_VALUE + 1)

HOG = {
    'orient': 16,
    'pix_per_cell': 4,
    'cells_per_block': 4,
    'colspace': 'YUV',
    # 'orient': 16,
    # 'pix_per_cell': 4,
    # 'cells_per_block': 6,
    # 'colspace': 'YCrCb',
    # pixels_per_cell=(4, 4),
    # cells_per_block=(6, 6),
}


def compute_avg_diff(points):
    if len(points) > 1:
        return np.mean([(it2-it1) for (it1,it2) in zip(points[0::2], points[1::2])])
    else:
        return 0


# Define a function to return some characteristics of the dataset
def get_data_stats(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    img = cv2.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = img.dtype
    # Return data_dict
    return data_dict


def print_data_stats(data_info):
    print('Found {n_cars} cars and {n_notcars} non-cars.'.format(**data_info))
    print('Image size: {image_shape}'.format(**data_info))
    print('Data type: {data_type}'.format(**data_info))


def assert_scale(image):
    max_val = np.max(image)
    assert max_val > 1, max_val


def extract_color_features(img, size=(16, 16)):
    return cv2.resize(img, size).ravel()


def extract_hist_features(img, nbins=32, bins_range=(0, 256)):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3,3))
    imcopy = np.copy(img)
    imcopy = cv2.cvtColor(imcopy, cv2.COLOR_RGB2LAB)
    channel1, channel2, channel3 = cv2.split(imcopy)
    channel1 = clahe.apply(channel1)
    channel2 = clahe.apply(channel2)
    channel3 = clahe.apply(channel3)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(channel1, bins=nbins, range=bins_range)
    channel2_hist = np.histogram(channel2, bins=nbins, range=bins_range)
    channel3_hist = np.histogram(channel3, bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def extract_hog_features(img,  vis=False, feature_vec=True, channel=0):
    imcopy = np.copy(img)
    imcopy = cv2.cvtColor(imcopy, getattr(cv2, "COLOR_RGB2%s" % HOG['colspace']))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3,3))
    _channel = clahe.apply(cv2.split(imcopy)[channel])

    ret = hog(
        _channel,
        orientations=HOG['orient'],
        pixels_per_cell=(HOG['pix_per_cell'], HOG['pix_per_cell']),
        cells_per_block=(HOG['cells_per_block'], HOG['cells_per_block']),
        block_norm= 'L2-Hys',
        transform_sqrt=True,
        visualise=vis,
        feature_vector=feature_vec
    )
    if not vis:
        return (ret, None)
    else:
        return ret


def extract_features(img, color=True, hist=True, hog=True, coltrans=False):
    imcopy = np.copy(img)
    features = []
    if coltrans:
        imcopy = cv2.cvtColor(imcopy.astype(np.uint8), cv2.COLOR_BGR2RGB)
    if color is True:
        color_features = extract_color_features(imcopy, size=(16, 16))
        features.append(color_features)
    if hist is True:
        hist_features = extract_hist_features(imcopy, nbins=16)
        features.append(hist_features)
    if hog is True:
        _get_hog = partial(
            extract_hog_features, imcopy, vis=False, feature_vec=False)

        hog_features = []
        hog1, hog1_vis = _get_hog(channel=0)
        hog_features.append(hog1.ravel())
        # hog2, hog1_vis = _get_hog(channel=1)
        # hog3, hog1_vis = _get_hog(channel=2)

        features.append(np.hstack(hog_features))
    return np.concatenate(features)


def save_data_sample(cars, notcars, fname='data_example.png'):
    fname = '%s/%s' % (OUTPUT_FOLDER, fname)
    # Just for fun choose random car / not-car indices and plot example images
    car_ind = np.random.randint(0, len(cars))
    notcar_ind = np.random.randint(0, len(notcars))

    # Read in car / not-car images
    car_image = cv2.imread(cars[car_ind])
    notcar_image = cv2.imread(notcars[notcar_ind])
    assert_scale(car_image)
    assert_scale(notcar_image)

    # Plot the examples
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(car_image)
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(notcar_image)
    plt.title('Example Not-car Image')
    fig.savefig(fname)
    print("%s saved." % fname)


def train_svc(no_samples=100):
    cars = glob.glob('dataset/vehicles/KITTI_extracted/*.png')
    notcars = glob.glob('dataset/non-vehicles/*/*.png')
    np.random.shuffle(cars)
    np.random.shuffle(notcars)

    data_info = get_data_stats(cars, notcars)
    print_data_stats(data_info)
    save_data_sample(cars, notcars)

    cars_features = []
    for path in cars[:no_samples]:
        cars_features.append(extract_features(cv2.imread(path), coltrans=True))

    notcars_features = []
    for path in notcars[:3000]:
        notcars_features.append(extract_features(cv2.imread(path), coltrans=True))

    features = np.vstack((cars_features, notcars_features)).astype(np.float64)
    labels = np.hstack((np.ones(len(cars_features)), np.zeros(len(notcars_features))))

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=rand_state)

    X_scaler = StandardScaler().fit(X_train)
    scaled_X_train = X_scaler.transform(X_train)
    scaled_X_test = X_scaler.transform(X_test)

    # parameters = {'kernel':('linear', 'poly', 'rbf'), 'gamma': ['auto', 0.1, 0.3, 0.9, 3, 10, 30, 100, 300, 1000], 'C':[0.1, 0.3, 0.9, 1, 3, 10, 30, 100, 300, 1000]}
    # svc = SVC()
    # clf = GridSearchCV(svc, parameters)
    # svc = clf.fit(scaled_X_train, y_train)

    # Use a linear SVC (support vector classifier)
    svc = SVC(kernel='rbf', gamma='auto', C=10.0, random_state=rand_state)
    # Train the SVC
    svc.fit(scaled_X_train, y_train)

    print('Test Accuracy of SVC = ', svc.score(scaled_X_test, y_test))
    return svc, X_scaler


def slide_window(img, svc=None, scaler=None, cells_per_step=None, window_scale=1, x_limits=[None, None], y_limits=[None, None]):
    start = time.time()

    x_start = 0 if x_limits[0] is None else x_limits[0]
    x_stop = img.shape[1] if x_limits[1] is None else x_limits[1]
    y_start = 0 if y_limits[0] is None else y_limits[0]
    y_stop = img.shape[0] if y_limits[1] is None else y_limits[1]

    roi = img[y_start:y_stop, x_start:x_stop, :]
    hog_area = cv2.resize(roi, (np.int(roi.shape[1]/window_scale), np.int(roi.shape[0] / window_scale)))
    _get_hog = partial(
        extract_hog_features, hog_area, vis=False, feature_vec=False)

    hog1, hog1_vis = _get_hog(channel=0)
    hog2, hog2_vis = _get_hog(channel=1)
    hog3, hog3_vis = _get_hog(channel=2)

    nyblocks, nxblocks = hog1.shape[:2]

    window_size = 64
    hog_window_size = (window_size // HOG['pix_per_cell']) - HOG['cells_per_block'] + 1
    scaled_window_size = np.int(window_size * window_scale)
    hog_x_windows = ((nxblocks - hog_window_size) // cells_per_step) + 1
    hog_y_windows = ((nyblocks - hog_window_size) // cells_per_step) + 1

    frame_heatmap = np.zeros(img.shape[:2])
    debug_image = np.copy(img)
    # print("Before loop: {:.2f}".format(time.time() - start))
    for idx_y in range(hog_y_windows):
        hog_y_pos = idx_y * cells_per_step
        y_pos = hog_y_pos * HOG['pix_per_cell']
        for idx_x in range(hog_x_windows):
            hog_x_pos = idx_x * cells_per_step
            x_pos = hog_x_pos * HOG['pix_per_cell']
            hog_bbox = (
                (hog_x_pos, hog_y_pos),
                (hog_x_pos+hog_window_size, hog_y_pos+hog_window_size)
            )
            bbox = (
                (x_start + np.int(x_pos * window_scale), y_start + np.int(y_pos * window_scale)),
                (x_start + np.int(x_pos * window_scale) + scaled_window_size, y_start + np.int(y_pos * window_scale) + scaled_window_size)
            )
            hog1_feat = hog1[hog_bbox[0][1]:hog_bbox[1][1],hog_bbox[0][0]:hog_bbox[1][0]].ravel()
            # hog2_feat = hog1[hog_bbox[0][1]:hog_bbox[1][1],hog_bbox[0][0]:hog_bbox[1][0]].ravel()
            # hog3_feat = hog1[hog_bbox[0][1]:hog_bbox[1][1],hog_bbox[0][0]:hog_bbox[1][0]].ravel()
            crop = img[bbox[0][1]:bbox[1][1],bbox[0][0]:bbox[1][0]]
            resize = cv2.resize(crop, (window_size, window_size))
            bbox_features = extract_features(resize, hog=False)
            features = np.concatenate([bbox_features, hog1_feat]).reshape(1, -1)
            scaled_features = scaler.transform(features)

            output = svc.predict(scaled_features)
            if output[0] == 1.0:
                frame_heatmap[
                    bbox[0][1]:bbox[1][1],
                    bbox[0][0]:bbox[1][0]
                ] += 1
                continue
                cv2.rectangle(debug_image, bbox[0], bbox[1], (250, 50, 50), 4)
                # cv2.rectangle(debug_image, bbox[0], bbox[1], (np.random.randint(100, 255), np.random.randint(100, 255), 50), 1)
            else:
                continue
                cv2.rectangle(debug_image, bbox[0], bbox[1], (50, 50, 250), 1)
    # print("Total time: {:.2f}".format(time.time() - start))
    return frame_heatmap, debug_image


def pipeline(state, slides, image):
    # image must be RGB
    assert_scale(image)
    state.setdefault('frameno', 0)

    copy_heatmap = np.copy(state['heatmap'])
    frame_heatmap = np.zeros(image.shape[:2])
    if state['frameno'] % HEATMAP_REFRESH_FREQ == 0:

        for slide in slides:
            frame_heatmap += slide(image)[0]

        # windows overlap threshold
        frame_heatmap[frame_heatmap < state['detections_threshold']] = 0

        # object detected => increase
        state['heatmap'][frame_heatmap > 0] += 1
        # object not detected => decrease
        state['heatmap'][frame_heatmap == 0] -= 1

    # increase frame number
    state['frameno'] += 1

    # reset negatives to 0 to allow objects to appear there
    state['heatmap'][state['heatmap'] < 0] = 0
    # reset to maximum
    state['heatmap'][state['heatmap'] > MAX_HEATMAP_VALUE] = MAX_HEATMAP_VALUE

    ret, thresh = cv2.threshold(state['heatmap'].astype(np.uint8), state['frames_threshold'], 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated_image = np.copy(image)
    for cnt in contours:
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        obj = None

        min_dist = None
        close_objects = []
        for _uuid, _obj in state['objects'].items():
            _dist = distance.euclidean((cx, cy), _obj['centroids'][-1])
            if _dist < OBJECT_MERGING_DISTANCE:
                close_objects.append(_uuid)
                if min_dist is None or min_dist > _dist:
                    obj = _obj
                    min_dist = _dist
                    obj['centroids'].append(
                        ((1-CENTROID_CHANGE_RATE) * obj['centroids'][-1] + CENTROID_CHANGE_RATE * np.array((cx, cy))).astype(np.int)
                    )

        for _uuid in close_objects:
            if obj['uuid'] != _uuid:
                del(state['objects'][_uuid])

        if obj is None:
            # new object detected
            _uuid = uuid.uuid4()
            obj = {
                'uuid': _uuid,
                'confidence': 0,
                'centroids': [np.array((cx, cy))],
                'color': [
                    np.random.randint(100, 255),
                    np.random.randint(100, 255),
                    np.random.randint(100, 255),
                ]
            }
            state['objects'][_uuid] = obj
        obj['increase_confidence'] = True
        height = np.int(np.sqrt(cv2.contourArea(cnt)) * 0.4)
        width = np.int(height * 1.3)
        curr_size = np.array([width, height])
        obj['size'] = (CENTROID_CHANGE_RATE * curr_size + (1-CENTROID_CHANGE_RATE) * obj.get('size', curr_size)).astype(np.int)
        # approx = cv2.approxPolyDP(cnt, epsilon=10, closed=True)
        # cv2.polylines(annotated_image, [approx], True, obj['color'], 3)
        # cv2.rectangle(annotated_image, approx[0][0], approx[1][0], (0, 190, 0), 3)

    for _uuid, _obj in state['objects'].items():
        if _obj['increase_confidence']:
            _obj['confidence'] = min(MAX_CONFIDENCE, max(1, _obj['confidence'] * 2))
            _obj['increase_confidence'] = False
        else:
            _obj['confidence'] -= 1
        _obj['confidence'] = 0 if _obj['confidence'] < 0 else _obj['confidence']
        bbox = [
            (_obj['centroids'][-1][0]-_obj['size'][0], _obj['centroids'][-1][1]-_obj['size'][1]),
            (_obj['centroids'][-1][0]+_obj['size'][0], _obj['centroids'][-1][1]+_obj['size'][1]),
        ]
        if _obj['confidence'] >= CONFIDENCE_THRESHOLD:
            mean_centroid = np.mean(np.array(_obj['centroids'][-50:]), axis=0).astype(np.int)
            (vx, vy, x, y) = cv2.fitLine(
                np.array([
                    mean_centroid,
                    (image.shape[1]//2, 430)
                ]),
                cv2.DIST_L2, 0, 0.01, 0.01)

            # top = int((-x*vy/vx) + y)
            # bottom = int(((image.shape[1] - x)*vy/vx) + y)

            line = np.vectorize(lambda X: vy/vx * (X - x) + y)

            avg_diff = np.int(compute_avg_diff(_obj['centroids'][-20:]))

            if avg_diff < 0:
                plotx = np.arange(_obj['centroids'][-1][0]+10*avg_diff, _obj['centroids'][-1][0])
            else:
                plotx = np.arange(_obj['centroids'][-1][0], _obj['centroids'][-1][0]+10*avg_diff)

            if  avg_diff != 0:
                ploty= line(plotx).astype(np.uint)

                corrected_x = np.int(_obj['centroids'][-1][0] + avg_diff)
                _obj['centroids'][-1] = np.array((corrected_x, np.int(line(corrected_x))))
                if avg_diff < 0:
                    cv2.line(annotated_image, (plotx[0], ploty[0]), (plotx[-1], ploty[-1]), (0, 255, 0), 2)
                else:
                    cv2.line(annotated_image, (plotx[0], ploty[0]), (plotx[-1], ploty[-1]), (255, 0, 0), 2)

            cv2.circle(
                annotated_image,
                (_obj['centroids'][-1][0], mean_centroid[0]),
                7,
                (255, 0, 0),
                thickness=-1,
            )
            cv2.rectangle(annotated_image, bbox[0], bbox[1], _obj['color'], 3)
            cv2.circle(
                annotated_image,
                # (_obj['centroids'][-1][0], _obj['centroids'][-1][1]),
                bbox[0],
                7,
                color=(0, _obj['confidence'] * CONFIDENCE_SCALER, 0),
                thickness=-1,
            )

    color_heatmap = np.dstack([copy_heatmap]*3) * VALUE_SCALER
    color_heatmap[:,:][copy_heatmap > 0] = color_heatmap[:,:][copy_heatmap > 0] * VALUE_SCALER
    color_heatmap[:,:][frame_heatmap > 0] = np.dstack([0, 50, 0])
    color_heatmap[:,:][(state['heatmap'] == 0) & (copy_heatmap > 0)] = np.dstack([50, 0, 0])
    plt.imsave('heatmap/frame-%s.png' % state['frameno'], color_heatmap)
    # plt.imsave('debug-heatmap.png', color_heatmap)
    # plt.imsave('debug.png', annotated_image)
    return annotated_image


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    subparsers = parser.add_subparsers()
    parser_train = subparsers.add_parser('train')
    parser_pipeline = subparsers.add_parser('pipeline')
    parser_video = subparsers.add_parser('video')

    parser_train.add_argument('--action', type=str, default='train')
    parser_pipeline.add_argument('--action', type=str, default='pipeline')
    parser_pipeline.add_argument('image', type=str)
    parser_video.add_argument('--action', type=str, default='video')
    parser_video.add_argument('video_in', type=str)
    parser_video.add_argument('video_out', type=str)
    parser_video.add_argument('--start', type=float, default=-1)
    parser_video.add_argument('--end', type=float, default=-1)

    arguments = parser.parse_args()

    if (
        arguments.action == 'train' or
        not (os.path.isfile('svc.p') and os.path.isfile('scaler.p'))
    ):
        svc, scaler = train_svc(no_samples=500)
        with open('svc.p', 'wb') as _f:
            pickle.dump(svc, _f)
        with open('scaler.p', 'wb') as _f:
            pickle.dump(scaler, _f)
        print('SVC saved')
    elif not arguments.action == 'train':
        with open('svc.p', 'rb') as _f:
            svc = pickle.load(_f)
        print('SVC loaded')
        with open('scaler.p', 'rb') as _f:
            scaler = pickle.load(_f)
        print('Scaler loaded')

    state = {
        'scaler': scaler,
        'svc': svc,
        'detections_threshold': 0,
        'frames_threshold': 0,
        'objects': {}
    }
    _slide_window = partial(
        slide_window,
        svc=state['svc'],
        scaler=state['scaler'],
    )
    slide1 = partial(
        _slide_window,
        window_scale=1.5,
        cells_per_step=7,
        y_limits=[400, 500],
    )
    slide2 = partial(
        _slide_window,
        window_scale=1.8,
        cells_per_step=5,
        y_limits=[400, 580],
    )
    if arguments.action == 'pipeline':
        image = cv2.imread(arguments.image)
        state['heatmap'] = np.zeros(image.shape[0:2])
        pipeline(state, cv2.cvtColor(image, cv2.COLOR_BGR2RGB), [slide1, slide2])
    elif arguments.action == 'video':
        # TODO set to 5
        state['frames_threshold'] = 1
        from moviepy.editor import VideoFileClip
        video_in = VideoFileClip(arguments.video_in)  # .subclip(35, 50)
        state['heatmap'] = np.zeros([video_in.size[1], video_in.size[0]])
        video_out = video_in.fl_image(partial(pipeline, state, [slide1, slide2]))
        video_out.write_videofile(arguments.video_out, audio=False)

if __name__ == "__main__":
    main()
