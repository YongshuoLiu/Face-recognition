from flask import Flask, render_template, request, send_from_directory
import os
import base64
# import cv2
import numpy as np
import requests
from zipfile import ZipFile

import os
import time
import json
import numpy as np
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision import utils as vutils
from network_files import MaskRCNN
from backbone import resnet50_fpn_backbone
from draw_box_utils import draw_objs
import imageio
from dnet.src.detector import detect_faces
from dnet.src.utils import show_bboxes


def create_model(num_classes, box_thresh=0.5):
    backbone = resnet50_fpn_backbone()
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()



num_classes = 90  # 不包含背景
box_thresh = 0.5
weights_path = "./save_weights/model_0.pth"
img_path = "./test/captured_image.png"
label_json_path = './coco91_indices.json'

# get devices
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

# create model
model = create_model(num_classes=num_classes + 1, box_thresh=box_thresh)

# load train weights
assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
weights_dict = torch.load(weights_path, map_location='cpu')
weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
model.load_state_dict(weights_dict)
model.to(device)

# read class_indict
assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
with open(label_json_path, 'r') as json_file:
    category_index = json.load(json_file)



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def usre_uoload():
    from flask import Flask, request, send_file
    from werkzeug.utils import secure_filename
    import os
    from zipfile import ZipFile
    UPLOAD_FOLDER = 'test'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # load image
    assert os.path.exists(img_path), f"{img_path} does not exits."
    original_img = Image.open(img_path).convert('RGB')

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init

        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        predictions = model(img.to(device))[0]
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))

        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()
        predict_mask = predictions["masks"].to("cpu").numpy()

        img = img.to("cpu").numpy()
        temp_img = img.squeeze()
        cropped_image = temp_img[:, int(predict_boxes[0][1]):int(predict_boxes[0][3]),
                        int(predict_boxes[0][0]):int(predict_boxes[0][2])]
        image = np.transpose(cropped_image, (1, 2, 0))  # 调整维度顺序
        crop_image = (image * 255).astype(np.uint8)  # 转换为8位整数类型

        # print(predict_boxes)
        temp_img = predict_mask.squeeze()
        cropped_image = temp_img[int(predict_boxes[0][1]):int(predict_boxes[0][3]),
                        int(predict_boxes[0][0]):int(predict_boxes[0][2])]
        crop_mask = np.where(cropped_image > 0.7, 1, 0).astype(np.uint8)
        crop_mask_temp = (crop_mask * 255).astype(np.uint8)  # 转换为8位整数类型
        img = Image.fromarray(crop_mask_temp)
        img.save("./test/image_mask.jpg")

        mask = np.expand_dims(crop_mask, axis=2)
        crop_mask = np.repeat(mask, 3, axis=2)
        image_crop_final = crop_image * crop_mask
        img = Image.fromarray(image_crop_final)
        img.save("./test/image_crop.jpg")

        print("====================================================================================================")
        img_test_dnet = Image.open("./test/image_crop.jpg")
        # print(type(img_test_dnet))
        bounding_boxes, landmarks = detect_faces(img_test_dnet)

        # print(landmarks)
        # print(bounding_boxes.shape, landmarks.shape)
        image = show_bboxes(img_test_dnet, bounding_boxes, landmarks)
        image.save("./test/image_landmark.jpg")

        predict_mask = np.squeeze(predict_mask, axis=1)  # [batch, 1, h, w] -> [batch, h, w]

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")


        plot_img = draw_objs(original_img,
                             boxes=predict_boxes,
                             classes=predict_classes,
                             scores=predict_scores,
                             masks=predict_mask,
                             category_index=category_index,
                             line_thickness=3,
                             font='arial.ttf',
                             font_size=20)

        plot_img.save("./test/image_result.jpg")


    # Create a ZipFile object
    with ZipFile('images.zip', 'w') as zipf:
        for filename in ['image_crop.jpg', 'image_landmark.jpg', 'image_result.jpg', 'image_mask.jpg', 'image_mask.jpg']:
            zipf.write(os.path.join('test', filename))

    return send_file('images.zip', mimetype='zip')


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8900,debug=True)
