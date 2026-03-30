import dlib
import numpy as np
from PIL import Image
import scipy.ndimage

class FFHQCrop:
    def __init__(self, output_size=1024, transform_size=4096, enable_padding=True, dlib_model_path='shape_predictor_68_face_landmarks.dat', quad_scale = 1.):
        self.output_size = output_size
        self.transform_size = transform_size
        self.enable_padding = enable_padding
        self.quad_scale = quad_scale

        # Load Dlib's face detector and the 68-point shape predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(dlib_model_path)
    
    def from_image_path(self, image_path):
        img = Image.open(image_path).convert('RGB')
        return self.__call__(img)
        
    def __call__(self, img):
        img_array = np.array(img)

        # Detect faces in the image
        dets = self.detector(img_array, 1)
        if len(dets) == 0:
            raise ValueError("No faces detected in the image.")

        # Assume the first detected face is the target
        d = dets[0]
        shape = self.predictor(img_array, d)
        lm = np.array([[p.x, p.y] for p in shape.parts()])

        # Calculate auxiliary vectors
        eye_left = np.mean(lm[36:42], axis=0)
        eye_right = np.mean(lm[42:48], axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        mouth_left = lm[48]
        mouth_right = lm[54]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_eye = eye_right - eye_left
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        x *= self.quad_scale
        y *= self.quad_scale
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # Shrink
        shrink = int(np.floor(qsize / self.output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(img.size[0] / shrink)), int(np.rint(img.size[1] / shrink)))
            img = img.resize(rsize, Image.Resampling.LANCZOS)
            quad /= shrink
            qsize /= shrink

        # Crop
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))),
                int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0),
                min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]
            
        # Pad
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))),
                int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0),
                max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        
        if self.enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            img[:,:,0]
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]),
                            1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
            img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]
        
        # Transform
        img = img.transform((self.transform_size, self.transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
        if self.output_size < self.transform_size:
            img = img.resize((self.output_size, self.output_size), Image.Resampling.LANCZOS)
            
        return img

if __name__ == "__main__":
    import sys
    import os

    import argparse
    import pandas as pd
    
    from tqdm.auto import tqdm
    
    parser = argparse.ArgumentParser(description='FFHQ Crop')
    parser.add_argument('-i', type=str, help='Input images folder')
    parser.add_argument('-s', type=float, default=1.1, help='Quad scale')
    parser.add_argument('-o', type=str, help='Output images folder')
    parser.add_argument('--dlib-model-path', type=str, default='shape_predictor_68_face_landmarks.dat', help='Dlib model path')
    
    args = parser.parse_args()
    
    os.makedirs(args.o, exist_ok=True)
    ffhq_crop = FFHQCrop(quad_scale=args.s, dlib_model_path=args.dlib_model_path)
    
    df_dict = []
    VALID_FILE_EXTENSIONS = ['jpg', 'jpeg', 'png', 'JPEG', 'JPG', 'PNG']
    
    bar = tqdm(total=len([file for root, dirs, files in os.walk(args.i) for file in files if file.split('.')[-1] in VALID_FILE_EXTENSIONS]))

    for root, dirs, files in os.walk(args.i):
        for file in files:
            bar.update(1)
            if file.split('.')[-1] in VALID_FILE_EXTENSIONS:
                # if ofile exists continue
                if os.path.exists(os.path.join(args.o, file)):
                    df_dict.append({'file': file, 'status': 'exists', 'input_path': os.path.join(root, file), 'output_path': os.path.join(args.o, file)})
                    continue
                try:
                    img = ffhq_crop.from_image_path(os.path.join(root, file))
                    img.save(os.path.join(args.o, file))
                    df_dict.append({'file': file, 'status': 'success', 'input_path': os.path.join(root, file), 'output_path': os.path.join(args.o, file)})
                    do_break = True
                except Exception as e:
                    df_dict.append({'file': file, 'status': 'error', 'input_path': os.path.join(root, file), 'output_path': None, 'error': str(e)})
        
    bar.close()
    
    print(df_dict)
    
    df = pd.DataFrame(df_dict)
    df.to_csv(os.path.join(args.o, 'ffhq_crop.csv'), index=False)
        