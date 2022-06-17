import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import librosa
from numpy.fft import fft, fftfreq, rfft, rfftfreq



def update(frame):
    data = np.random.rand(20)
    opacity = np.logspace(0, -1, 30)

    # line = ax.stem(xs, ys*frame*5, data)

    # return ax.stem(xs, ys * frame, data)
    # return ax.bar(xs, frame*ys, data, zdir='y', color='r', alpha=np.e**-frame)
    return ax.bar(xs, data, frame*ys*5, zdir='y', color='r', alpha=opacity[frame])



    # # Shift all data to the right
    # data[:, 1:] = data[:, :-1]
    #
    # # Fill-in new values
    # data[:, 0] = np.random.uniform(0, 1, len(data))
    #
    # # Update data
    # for i in range(len(data)):
    #     lines[i].set_ydata(i + G * data[i])
    #
    # # Return modified artists
    # return lines


if __name__ == '__main__':
    # Fixing random state for reproducibility
    # np.random.seed(19680801)

    signal, s_rate = librosa.load('/Users/youngwang/Desktop/audio-visualization/bird.wav')
    duration = librosa.get_duration(y=signal, sr=s_rate)

    # for i in range(5):
    length = 10000
    frame = (signal[0: length])
    yf = np.abs(rfft(frame))
    # xf = rfftfreq(len(frame), 1 / s_rate)

    # y, sr = librosa.load('bird.mp3')
    # Create new Figure with black background
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Generate random data
    xs = np.arange(20)
    # ys = np.random.rand(20)
    zs = np.ones(len(xf))
    # Generate line plots

    # line = ax.stem(xs, ys, zs)
    line = ax.bar(xf, yf, zs, zdir='y', color='r')

    # lines = []
    # for i in range(len(data)):
    #     # Small reduction of the X extents to get a cheap perspective effect
    #     xscale = 1 - i / 200.
    #     # Same for linewidth (thicker strokes on bottom)
    #     lw = 1.5 - i / 100.0
    #     line, = ax.plot(xscale * X, i + G * data[i], color="w", lw=lw)
    #     lines.append(line)

    # Set y limit (or first line is cropped because of thickness)
    ax.set_ylim(0, 100)

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
    # frameno = 20
    # opacity = np.logspace(0, -1, frameno)
    # # Construct the animation, using the update function as the animation director.
    # anim = animation.FuncAnimation(fig, update, frameno, interval=20, repeat=False)
    # anim.save('../bird.gif', writer='PillowWriter', fps=60)

    plt.show()