# A simple script for ensembling a nose detector + a mask detector
# can be run on single image or video frames
# created by Shiqin on April 9, 2021
import cv2
import os
from PIL import Image
import argparse
from run_on_image import run_on_image
from run_on_video import run_on_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-mode', type=int, default=0, help='set 1 to run on image, 0 to run on video.')
    parser.add_argument('--img-path', type=str, help='path to an input image.')
    parser.add_argument('--imgs-path', type=str, help='path to directory of input images', default='./images')
    parser.add_argument('--video-path', type=str, default='0', help='path to your video, `0` means to use camera.')
    parser.add_argument('--show-results', action="store_true", help='whether to show the resulting images')
    parser.add_argument('--save-results', action="store_true" , help='whether to save the resulting images to output dir')
    args = parser.parse_args()

    if args.img_mode:
        show_results = bool(args.show_results)
        save_results = bool(args.save_results)
        if args.img_path:
            imgPath = args.img_path
            img = cv2.imread(imgPath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            output_info, image = run_on_image(img, input_size=(260, 260))
            if show_results:
                Image.fromarray(image).show()
            if save_results:
                Image.fromarray(image).save(f"output/{imgPath}")
        if args.imgs_path:
            for img_name in os.listdir(args.imgs_path):
                imgPath = args.imgs_path + img_name
                img = cv2.imread(imgPath)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                output_info, image = run_on_image(img, input_size=(260, 260))
                if show_results:
                    Image.fromarray(image).show()
                if save_results:
                    Image.fromarray(image).save(f"output/{imgPath}")

    else:
        video_path = args.video_path
        if args.video_path == '0':
            video_path = 0
        run_on_video(video_path, '', conf_thresh=0.5)
