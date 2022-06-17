import matplotlib.pyplot as plt
import matplotlib.animation as animation
import librosa
from numpy.fft import fft, fftfreq, rfft, rfftfreq

from numpy import fft
import numpy as np

# #
def update(frame):
    xf, yf = getframe(signal, s_rate, no_i =frame)
    # frame = (signal[s_rate * frame: s_rate * (frame + 1)])
    # yf = rfft(frame)
    # xf = rfftfreq(len(frame), 1 / s_rate)
    # plt.plot(xf, np.abs(yf))
    # plt.show()
    #

    # data = np.random.rand(20)
    opacity = np.logspace(0, -1, 30)

    # line = ax.stem(xs, ys*frame*5, data)
    #
    # return ax.stem(xs, ys * frame, data)
    # return ax.bar(xs, frame*ys, data, zdir='y', color='r', alpha=np.e**-frame)
    # return ax.bar(xf, yf, zs*frame, zdir='y',color='r', alpha=opacity[frame])
    return ax.bar(xf, yf, frame*zs, zdir='y', color='r', alpha=opacity[frame])
    # return ax.stem(xf, zs*frame, yf)



def getframe(signal, s_rate, no_i):

    frame = (signal[100*no_i: 100*(no_i+1)])
    yf = np.abs(rfft(frame))
    xf = rfftfreq(len(frame), 1 / s_rate)
    # plt.plot(xf, yf)
    # plt.show()

    # Return modified artists
    return xf,yf

# import scipy.io.wavfile as wavfile
# import scipy
# import scipy.fftpack as fftpk
# import numpy as np
# from matplotlib import pyplot as plt
#
# s_rate, signal = wavfile.read("tibet.wav")
#
# FFT = abs(scipy.fft(signal))
# freqs = fftpk.fftfreq(len(FFT), (1.0/s_rate))
#
# plt.plot(freqs[range(len(FFT)//2)], FFT[range(len(FFT)//2)])
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Amplitude')
# plt.show()
if __name__ == '__main__':
    # Fixing random state for reproducibility
    # np.random.seed(19680801)

    signal, s_rate = librosa.load('/Users/youngwang/Desktop/audio-visualization/bird.wav')
    duration = librosa.get_duration(y=signal, sr=s_rate)

    # for i in range(5):
    length = 50
    frame = (signal[0: length])
    yf = np.abs(rfft(frame))
    xf = rfftfreq(len(frame), 1 / s_rate)
    #     plt.plot(xf, yf)
    #     plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    zs = np.ones(len(xf))
    #
    # Generate random data
    # xs = xf
    # # ys = np.random.rand(20)
    # ys = np.ones_like(len(xf))
    # zs = yf
    # # Generate line plots

    # line = ax.stem(xf, zs, yf)

    line = ax.bar(xf, yf, zs, zdir='y', color='r')

    #
    # # lines = []
    # # for i in range(len(data)):
    # #     # Small reduction of the X extents to get a cheap perspective effect
    # #     xscale = 1 - i / 200.
    # #     # Same for linewidth (thicker strokes on bottom)
    # #     lw = 1.5 - i / 100.0
    # #     line, = ax.plot(xscale * X, i + G * data[i], color="w", lw=lw)
    # #     lines.append(line)
    #
    # # Set y limit (or first line is cropped because of thickness)
    # No ticks
    ax.set_xlabel('Loudness')
    ax.set_ylabel('Frequency')
    # ax.set_zlabel('Z')
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.set_zticks([])


    # # # 2 part titles to get different font weights
    # # ax.text(0.5, 1.0, "MATPLOTLIB ", transform=ax.transAxes,
    # #         ha="right", va="bottom", color="w",
    # #         family="sans-serif", fontweight="light", fontsize=16)
    # # ax.text(0.5, 1.0, "UNCHAINED", transform=ax.transAxes,
    # #         ha="left", va="bottom", color="w",
    # #         family="sans-serif", fontweight="bold", fontsize=16)
    # import librosa
    frameno = 30
    opacity = np.logspace(0, -1, frameno)
    # Construct the animation, using the update function as the animation director.
    anim = animation.FuncAnimation(fig, update, frameno, interval=20, repeat=False)
    # anim.save('../bird.gif', writer='PillowWriter', fps=60)

    plt.show()