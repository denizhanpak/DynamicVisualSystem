from scipy.ndimage import gaussian_filter
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
import argparse

parser = argparse.ArgumentParser(description='Applies a greyscale difference filter to a video.')
parser.add_argument('video_file', type=str, nargs=1,
                    help='/path/to/video.extension')
parser.add_argument('--threshold', type=float, nargs=1,
                    help='Threshold is the difference above which pixels will be marked. Higher values will make the retina less sensitive (default = 0.5).',default=[0.5])
parser.add_argument('--sigma', type=float, nargs=1,
                    help='Sigma is the standard deviation for the Gaussian blur applied to the original images. Higher values will make objects more uniform (default = 0.01)', default=[0.01])
parser.add_argument('--max_frame', type=int, nargs=1,
                    help='If set it is the maximum number of frames which will be calculated', default=[-1])
parser.add_argument('--out_file', type=str, nargs=1,
        help='/path/to/output_name (.mp4) extension is added automatically (default = output)', default=["output"])
parser.add_argument('--fps', type=int, nargs=1,
                    help='Frames per second of new video (default = 20)', default=[20])
args = parser.parse_args()

class Retina:
    def __init__(self, axis=2, threshold=0.5, sd=0.01):
        self.threshold = threshold
        self.sd = sd
        self.axis = axis

    def create_grayscale(self, image):
        rv = image.sum(axis=self.axis)/3 * 255
        return np.expand_dims(rv,axis=self.axis)

    def blur(self, image):
        return gaussian_filter(image, sigma=self.sd)

    
    def quantize(self, diff):
        diff[np.abs(diff) <= self.threshold] = 0
        diff[diff > self.threshold] = 1
        diff[diff < -self.threshold] = -1
        return diff

    def see(self, current_obs, prev_obs):
        current = self.create_grayscale(current_obs)
        prev = self.create_grayscale(prev_obs)
        current = self.blur(current)
        prev = self.blur(prev)
        dc = self.quantize(current - prev)
        return dc
    
    def next_index(self, n):
        return (n + 1) % 2
    
    # First set up the figure, the axis, and the plot element we want to animate
    def create_video(self,obs,vmin=0,vmax=255):
        fig = plt.figure()
        plt.axis('off')
        img = plt.imshow(obs[0],cmap='gray', vmin=vmin, vmax=vmax)
        frame_text = plt.text(2, 2, "Frame 0")

        def animate(i):
            img.set_array(obs[i])
            frame_text.set_text(f"Frame {i}")
            return img,
        anim = animation.FuncAnimation(fig, animate,
                                       interval=30,
                                       frames=obs.shape[0],
                                       blit=True)
        return anim
    
    def watch(self, video, outfile="im", fps=20, max_frame=-1):
        cap = cv2.VideoCapture(video)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        buff = np.empty((2, frameHeight, frameWidth, 3), np.dtype('uint8'))
        
        flag, buff[0] = cap.read()
        flag, buff[1] = cap.read()
        v = []
        index = 1
        timer = 0
        while flag:
            index = self.next_index(index)
            video = self.see(buff[index], buff[self.next_index(index)])
            video = np.uint8(video)
            v.append(video)
            try:
                flag, buff[index] = cap.read()
            except:
                break
            timer += 1
            if max_frame == timer:
                break
        cap.release()
        cv2.destroyAllWindows()
        anim = self.create_video(np.uint8(v),vmin=0,vmax=255)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(outfile + ".mp4", writer=writer)
        return anim

r = Retina(threshold=args.threshold[0],sd=args.sigma[0])
print(args.video_file)
r.watch(args.video_file[0],max_frame=args.max_frame[0], outfile=args.out_file[0],fps=args.fps[0])
