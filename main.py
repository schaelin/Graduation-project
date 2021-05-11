import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import signal
from scipy.signal import butter, filtfilt
import csv

band = (42, 180)
n = 7

def filter_bandpass(arr, fps, band):
    nyq = 60 * fps / 2
    coefficients = signal.butter(5, [band[0] / nyq, band[1] / nyq], 'bandpass')
    return signal.filtfilt(*coefficients, arr)


def detrend_signal(arr, win_size):
    if not isinstance(win_size, int):
        win_size = int(win_size)
    length = len(arr)
    norm = np.convolve(np.ones(length), np.ones(win_size), mode='same')
    mean = np.convolve(arr, np.ones(win_size), mode='same') / norm
    return (arr - mean) / mean


def estimate_average_pulserate(window_size, arr, srate):
    pad_factor = max(1, 60 * srate / window_size)
    n_padded = int(len(arr) * pad_factor)

    f, pxx = signal.periodogram(arr, fs=srate, window='hann', nfft=n_padded)
    max_peak_idx = np.argmax(pxx)
    return int(f[max_peak_idx] * 60)

'''
def calculate_value(img):
    b, g, r = cv2.split(img)
    value = np.mean(g)

    return value
'''

def calculate_value(img):
    img_cbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    height = img_cbcr.shape[0]
    width = img_cbcr.shape[1]
    img_y, img_cr, img_cb = cv2.split(img_cbcr)
    value_cb_center = np.mean(img_cb)
    value_cr_center = np.mean(img_cr)
    """
    for i in range(height):
        for j in range(width):
            cr = img_cr[i][j]
            cb = img_cb[i][j]
            img_cr[i][j] = value_cr_center + (cr - value_cr_center) * n
            img_cb[i][j] = value_cb_center + (cb - value_cb_center) * n

            if img_cr[i][j] > 255:
                img_cr[i][j] = 255
            if img_cb[i][j] > 255:
                img_cb[i][j] = 255
    """
    # return img_cb, img_cr
    return value_cb_center, value_cr_center


def maxvalue(arr1, arr2):
    value = np.argmax(arr1)
    value = arr2[value]
    return value


def filter_butterworth_bandpass(arr, srate, length, band, order=5):
    try:
        (minFreq, maxFreq) = band
        nyq = srate / 2.0
        n = len(arr)
        pad_factor = max(1, 60 * srate / length)
        n_padded = int(n * pad_factor)
        padded = np.zeros(n_padded)
        padded[:n] = arr[:]

        filter = butter(order, [minFreq / nyq, maxFreq / nyq], 'bandpass')
        bandpassed = filtfilt(*filter, padded)
        bandpassed = bandpassed[:n]
        return bandpassed

    except ValueError:
        return []


def main():
    # cap = cv2.VideoCapture(0)
    frame_01 = 0
    dir_path = 'C:/Users/82102/Desktop/t/Lee_data_26'
    data_list = os.listdir(dir_path)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontface.xml')
    # pulses = []
    # frequencies = []
    signals = []
    a = len(data_list)
    for data_index in range(130, a+1):
        data_name = dir_path + '/' + data_list[data_index]
        cap = cv2.VideoCapture(data_name)

        # signals = []

        i = 0
        is_detected = False
        x, y, w, h = 0, 0, 0, 0
        while i < 10 * 30:
            ok, frame = cap.read()


            #frame2 = cv2.resize(frame1, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 교수님 데이터 적용 시

            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if is_detected:
                pass
            else:
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    max = 0
                    # find bigger face
                    for i in range(1, (len(faces))):
                        if i == 1:
                            if faces[i - 1][2] > faces[i][2]:
                                max = i - 1
                            else:
                                max = i
                        else:
                            if faces[i][2] > faces[max][2]:
                                max = i
                        i += 1
                    x, y, w, h = faces[max]
                    is_detected = True

            if w != 0:
                # face = frame[y+int(0.2*h):(y+int(0.8*h)), x+int(0.2*w):x+int(0.8*w)]
                face = frame[y:y + h, x:x + w]

                # cv2.imshow('face', face)
                """
                value_cb, value_cr = calculate_value(face)
                value_cb = np.mean(value_cb)
                value_cr = np.mean(value_cr)
                value = (value_cb + value_cr) / 2
                """
                value_cb, value_cr = calculate_value(face)
                value = (value_cb + value_cr) / 2
                signals.append(value)

                # cv2.rectangle(frame, (x+int(0.2*w),y+int(0.2*h)), (x+int(0.8*w), (y+int(0.8*h))), (0,0,255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                frame_01 = cv2.resize(frame, dsize=(480, 640))

            cv2.imshow('frame', frame_01)
            key = cv2.waitKey(1)
            if key == ord('x'):
                break

            i += 1

        det = detrend_signal(signals, 30)
        filtered = filter_butterworth_bandpass(det, 15, 900, (0.7, 3.0))
        filtered_v2 = filter_bandpass(det, 30, band)
        f, P = signal.periodogram(filtered, fs=30.0)
        pulserate_v2 = estimate_average_pulserate(900, filtered_v2, 30)

        # pulses.append(filtered)
        # frequencies.append(P)

        print("당신의 심박수는 %d bpm 입니다." % pulserate_v2)
        # print(filtered.shape)

        """
        plt.subplot(221)
        plt.plot(signals)
        plt.subplot(222)
        plt.plot(det)
        plt.subplot(223)
        plt.plot(filtered)
        plt.subplot(224)
        plt.xlim(0.0, 7,0)
        plt.plot(f,P)
        plt.show()
        """

        cap.release()
        cv2.destroyAllWindows()

        import csv
        f = open('per_lee.csv','a', newline = '')
        wr = csv.writer(f)
        wr.writerow(filtered)
        f.close()

        f = open('fre_lee.csv', 'a', newline='')
        wr = csv.writer(f)
        wr.writerow(P)
        f.close()

        signals.clear()

if __name__ == '__main__':
    main()
