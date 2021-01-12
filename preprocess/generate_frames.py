import os
import cv2
import click

def get_video_duration(vidcap):
    fps = vidcap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    return duration*1000

def get_folder_name(filename):
    to_folder = filename.split('/')
    to_folder[-2] = 'frames_padding'
    to_folder = '/'.join(to_folder)
    if not os.path.isdir(to_folder):
        os.mkdir(to_folder)
    return to_folder

def get_frames(filename):
    vidcap = cv2.VideoCapture(filename)
    to_folder = get_folder_name(filename)
    count = 0
    duration = get_video_duration(vidcap)
    video_left = 10*1000 - duration # Videos are supposed to last 10 sec
    first_image=None
    while True:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, count*100)
        success,image = vidcap.read()
        if first_image is None:
            first_image=image
        if not success:
            break
        cv2.imwrite(os.path.join(to_folder,"{:06d}.png".format(int(video_left/100) + count)), image)     # save frame as JPEG file
        count += 1
    for i in range(int(video_left/100)):
        cv2.imwrite(os.path.join(to_folder,"{:06d}.png".format(i)), first_image)     # first images (black) will be set like the first one

    print("{} images are extacted in {}.".format(count,to_folder))

@click.command()
@click.option('--folder', help='Folder where the mp4 videos are')
def main(folder):
    full_name_files = [os.path.join(folder, file) for file in os.listdir(folder)]
    for idx, filename in enumerate(full_name_files[1771:]):
        get_frames(filename)
        if not idx % 50:
            print('Left {}'.format(len(full_name_files) - idx))
    return 0

if __name__ == '__main__':
    main()
