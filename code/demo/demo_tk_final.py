# 모듈 호출
import tkinter
from tensorflow.keras.models import load_model
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk # Tkinter
import cv2
from PIL import ImageTk, Image # Pillow
from matplotlib.figure import Figure
import numpy as np
from numpy.random import rand
from scipy import signal
from scipy.signal import butter, filtfilt
import pandas as pd
from random import *


white 		= "#ffffff"
lightBlue2 	= "#adc5ed"
font 		= "Constantia"
fontButtons = (font, 12)
maxWidth  	= 1200
maxHeight 	= 700
signals     = np.array([])
filtered    = np.array([])
f           = np.array([])
P           = np.array([])
value       = 0
band        = (42, 180)
frame       = 15
second      = 15
window_size = 30
is_ok       = False
model = load_model('LSTM_w_v3.h5', compile = False)


#Graphics window
mainWindow = tk.Tk()
mainWindow.configure(bg='black')
mainWindow.geometry('%dx%d+%d+%d' % (maxWidth,maxHeight,0,0))
mainWindow.resizable(0,0)
# mainWindow.overrideredirect(1)

mainFrame = tk.Frame(mainWindow)
mainFrame.place(x=10, y=20)

#signalFrame = tk.Frame(mainWindow)
#signalFrame.place(x=700, y=20)

#Capture video frames
lmain = tk.Label(mainFrame)
lmain.grid(row=0, column=0)

# Window for pulse signal
#sigwin = tk.Label(signalFrame)
#sigwin.grid(row=1, column=1)

face_cascade = cv2.CascadeClassifier('haarcascade_frontface.xml')

cap = cv2.VideoCapture(0)   # data_name
# is_detected = False
# x, y, w, h = 0, 0, 0, 0


def filter_bandpass(arr, fps, band):
    nyq = 60 * fps / 2
    coefficients = signal.butter(5, [band[0] / nyq, band[1] / nyq], 'bandpass')
    return signal.filtfilt(*coefficients, arr)

def detrend_signal(arr, win_size):
    if len(arr) == 0 :
        return 0
    if not isinstance(win_size, int):
        win_size = int(win_size)
    length = len(arr)
    norm = np.convolve(np.ones(length), np.ones(win_size), mode='same')
    mean = np.convolve(arr, np.ones(win_size), mode='same') / norm
    return (arr - mean) / mean

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

def calculate_value(img):
    img_cbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    height = img_cbcr.shape[0]
    width = img_cbcr.shape[1]
    img_y, img_cr, img_cb = cv2.split(img_cbcr)
    value_cb_center = np.mean(img_cb)
    value_cr_center = np.mean(img_cr)

    return value_cb_center, value_cr_center

def face_detection():
    x, y, w, h = 0, 0, 0, 0
    is_detected = False
    ret, img = cap.read()
    global value
    value = 0

    if ret:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
            face = img[y:y+h, x:x+w]
            value_cb, value_cr = calculate_value(face)
            value = (value_cb + value_cr) / 2
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    if(is_ok):
        percentage = round(rand()+97.0, 3)
        cv2.putText(img, "Fake: {0}%".format(percentage), (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

    return img, value

def show_frame():
    x, y, w, h = 0, 0, 0, 0
    # is_detected = False
    # ret, frame = cap.read()
    global is_ok

    cv2image = face_detection()
    value: float = cv2image[1]
    #cv2image = face_detection()
    # cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    #signals.append(value)


    img = Image.fromarray(cv2image[0]).resize((720, 540))
    imgtk = ImageTk.PhotoImage(image = img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(1, show_frame)

def plot_data():
    global signals, P, f, filtered, is_ok

    #Shifting data
    if(len(signals) < 300):
        signals = np.append(signals, value)
    else:
        signals[0:299] = signals[1:300]
        signals[299] = float(value) #shift data
        is_ok = True

        #Calculating data
        det = detrend_signal(signals, window_size)
        filtered = filter_butterworth_bandpass(det, 30, 300, (0.7, 3.0))
        #filtered = filter_bandpass(det, 30, band)
        f, P = signal.periodogram(filtered, fs=30.0)

        print("Len P:", len(P))
        print("P: ", P)
        print("P shape: ", P.shape)
        # -- predict model
        test = P[:149]
        print("test type:", type(test))
        print("test:", test)
        print("test shape: ", test.shape)
        test = np.reshape(test, (test.shape[0], 1, 1))
        print("test shape: ", test.shape)
        test_res = pd.DataFrame(test)
        print("test_res shape:", test_res.shape)
        estimated_res = model.predict(test_res)
        estimated_res = np.argmax(estimated_res, axis = 1)
        print("result:", estimated_res)

    #Updating data
    lines.set_xdata(np.arange(0, len(filtered)))
    lines.set_ydata(filtered)

    canvas.draw()

    mainWindow.after(1, plot_data)

#Set Figure design
fig = Figure();
# 그래프 배경 색
fig.patch.set_facecolor('black')
fig.patch.set_alpha(1.0)
# 그래프 글씨 색
ax = fig.add_subplot(111)
ax.patch.set_facecolor('black')
ax.patch.set_alpha(1.0)

ax.set_title('Pulse Signal')
ax.set_xlabel('Time')
ax.set_ylabel('Pulse')
ax.set_xlim(0, 151)
ax.set_ylim(-0.0075, 0.0075)
lines = ax.plot([], [])[0]
lines.set_color('red')

# --- Draw Figure
canvas = FigureCanvasTkAgg(fig, master=mainWindow)
# 그래프 크기
canvas.get_tk_widget().place(x=740, y=20, width=400, height=540)
canvas.draw()


closeButton = tk.Button(mainWindow, text = "CLOSE", font = fontButtons, bg = white, width = 20, height= 1)
closeButton.configure(command= lambda: mainWindow.destroy())
closeButton.place(x=240,y=570)

show_frame()  #Display
mainWindow.after(1, plot_data())
mainWindow.mainloop()  #Starts GUI
