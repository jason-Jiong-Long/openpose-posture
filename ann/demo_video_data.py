import copy
import numpy as np
import cv2
from glob import glob
import os
import argparse
import json

# video file processing setup
# from: https://stackoverflow.com/a/61927951
import argparse
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class FFProbeResult(NamedTuple):
    return_code: int
    json: str
    error: str


def ffprobe(file_path) -> FFProbeResult:
    command_array = ["ffprobe",
                     "-v", "quiet",
                     "-print_format", "json",
                     "-show_format",
                     "-show_streams",
                     file_path]
    result = subprocess.run(command_array, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    return FFProbeResult(return_code=result.returncode,
                         json=result.stdout,
                         error=result.stderr)

# openpose setup
from src import model
from src import util
from src.body import Body
from src.hand import Hand

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')
data_name=973#初始資料名稱編碼

def process_frame(frame, body=True, hands=True):
    global data_name
    canvas = copy.deepcopy(frame)
    if body:
        candidate, subset = body_estimation(frame)
        canvas = util.draw_bodypose(canvas, candidate, subset)

        for a in range(subset.shape[0]):
            data_dict={}
            #print(a)
            for b in range(18):
                for c in range(candidate.shape[0]):
                    if subset[a,b]==int(candidate[c,3]):
                        data_dict.setdefault(b,[candidate[c,0],candidate[c,1],candidate[c,2]])

                    if subset[a,b]==-1:
                        data_dict.setdefault(b,[-1,-1,-1])


            data_list1=[]
            data_list2=[]
            data_list3=[]
            data_list4=[]
            data_list5=[]
            data_list6=[]
            data_list7=[]
            data_list8=[]
            data_list9=[]
            data_list10=[]
            data_list11=[]
            data_list12=[]
            data_list13=[]
            data_list14=[]
            data_list15=[]
            data_list16=[]
            data_list17=[]
            data_list18=[]

            for d in range(6):
                data_list1.append(data_dict[d][0])
            for e in range(6):
                data_list2.append(data_dict[e+6][0])
            for f in range(6):
                data_list3.append(data_dict[f+12][0])
            for g in range(6):
                data_list4.append(data_dict[g][1])
            for h in range(6):
                data_list5.append(data_dict[h+6][1])
            for i in range(6):
                data_list6.append(data_dict[i+12][1])
            data_lists1=np.vstack([data_list1,data_list2,data_list3,data_list4,data_list5,data_list6])#x,y

            for j in range(6):
                data_list7.append(data_dict[j][1])
            for k in range(6):
                data_list8.append(data_dict[k+6][1])
            for l in range(6):
                data_list9.append(data_dict[l+12][1])
            for m in range(6):
                data_list10.append(data_dict[m][2])
            for n in range(6):
                data_list11.append(data_dict[n+6][2])
            for o in range(6):
                data_list12.append(data_dict[o+12][2])
            data_lists2=np.vstack([data_list7,data_list8,data_list9,data_list10,data_list11,data_list12])#y,sorce

            for p in range(6):
                data_list13.append(data_dict[p][2])
            for q in range(6):
                data_list14.append(data_dict[q+6][2])
            for r in range(6):
                data_list15.append(data_dict[r+12][2])
            for s in range(6):
                data_list16.append(data_dict[s][0])
            for t in range(6):
                data_list17.append(data_dict[t+6][0])
            for u in range(6):
                data_list18.append(data_dict[u+12][0])
            data_lists3=np.vstack([data_list13,data_list14,data_list15,data_list16,data_list17,data_list18])#sorce,x
            data_list_all=np.stack((data_lists1,data_lists2,data_lists3),axis=0)
            print(data_dict)
            print(data_lists1)
            np.save('./dataset_ann/2/tensor_data'+str(data_name)+'.npy', data_lists1)
            data_name+=1
    if hands:
        hands_list = util.handDetect(candidate, subset, frame)
        all_hand_peaks = []
        for x, y, w, is_left in hands_list:
            peaks = hand_estimation(frame[y:y+w, x:x+w, :])
            peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
            peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
            all_hand_peaks.append(peaks)
        canvas = util.draw_handpose(canvas, all_hand_peaks)
    return canvas

# writing video with ffmpeg because cv2 writer failed
# https://stackoverflow.com/questions/61036822/opencv-videowriter-produces-cant-find-starting-number-error
import ffmpeg
import math
# open specified video
parser = argparse.ArgumentParser(
        description="Process a video annotating poses detected.")
parser.add_argument('file', type=str, help='Video file location to process.')
parser.add_argument('--no_hands', action='store_true', help='No hand pose')
parser.add_argument('--no_body', action='store_true', help='No body pose')
args = parser.parse_args()
video_file = args.file
cap = cv2.VideoCapture(video_file)

# get video file info
print(args.file)
ffprobe_result = ffprobe(args.file)
#print(ffprobe_result)
info = json.loads(ffprobe_result.json)

videoinfo = [i for i in info["streams"] if i["codec_type"] == "video"][0]
input_fps = videoinfo["avg_frame_rate"]
# input_fps = float(input_fps[0])/float(input_fps[1])
input_pix_fmt = videoinfo["pix_fmt"]
input_vcodec = videoinfo["codec_name"]

# define a writer object to write to a movidified file
postfix = info["format"]["format_name"].split(",")[0]
output_file = ".".join(video_file.split(".")[:-1])+".processed." + postfix


class Writer():
    def __init__(self, output_file, input_fps, input_framesize, input_pix_fmt,
                 input_vcodec):
        if os.path.exists(output_file):
            os.remove(output_file)
        self.ff_proc = (
            ffmpeg
            .input('pipe:',
                   format='rawvideo',
                   pix_fmt="bgr24",
    #               s='%sx%s'% ((math.ceil(input_framesize[1])*2)*0.5,(math.ceil(input_framesize[0])*2)*0.5)),
                   s= '%sx%s' % (input_framesize[1],input_framesize[0]),
                   r=input_fps)
            .output(output_file, pix_fmt=input_pix_fmt, vcodec=input_vcodec)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    def __call__(self, frame):
        self.ff_proc.stdin.write(frame.tobytes())

    def close(self):
        self.ff_proc.stdin.close()
        self.ff_proc.wait()


writer = None
while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
        break

    posed_frame = process_frame(frame, body=not args.no_body,
                                       hands=not args.no_hands)

    if writer is None:
        input_framesize = posed_frame.shape[:2]
        writer = Writer(output_file, input_fps, input_framesize, input_pix_fmt,
                        input_vcodec)

    cv2.imshow('frame', posed_frame)

    # write the frame
    writer(posed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.close()
cv2.destroyAllWindows()
