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
from dnet.src.detector import Dnet
from dnet.src.utils import show_bboxes
#import sys
#sys.path.insert(0, '../facenet')
#from facenet_torch.nets.inception_resnetv1 import InceptionResnetV1
#from facenet_torch.nets.mobilenet import MobileNetV1

from facenet_torch.facenet import Facenet
#from facenet_torch.nets.facenet import Facenet as facenet
#from facenet_torch.utils.utils import preprocess_input, resize_image, show_config

import cv2
import math


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
detect_faces_model = Dnet()
face_model = Facenet()


# read class_indict
assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
with open(label_json_path, 'r') as json_file:
    category_index = json.load(json_file)



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

#@app.route('/chechin')
#def index():
 #   return render_template('index.html')

def Alignment_1(img,landmark):
    if landmark.shape[0]==68:
        x = landmark[36,0] - landmark[45,0]
        y = landmark[36,1] - landmark[45,1]
    elif landmark.shape[0]==5:
        x = landmark[0,0] - landmark[1,0]
        y = landmark[0,1] - landmark[1,1]

    if x==0:
        angle = 0
    else:
        angle = math.atan(y/x)*180/math.pi

    center = (img.shape[1]//2, img.shape[0]//2)

    RotationMatrix = cv2.getRotationMatrix2D(center, angle, 1)
    new_img = cv2.warpAffine(img,RotationMatrix,(img.shape[1],img.shape[0]))

    RotationMatrix = np.array(RotationMatrix)
    new_landmark = []
    for i in range(landmark.shape[0]):
        pts = []
        pts.append(RotationMatrix[0,0]*landmark[i,0]+RotationMatrix[0,1]*landmark[i,1]+RotationMatrix[0,2])
        pts.append(RotationMatrix[1,0]*landmark[i,0]+RotationMatrix[1,1]*landmark[i,1]+RotationMatrix[1,2])
        new_landmark.append(pts)

    new_landmark = np.array(new_landmark)

    return new_img, new_landmark

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


        bounding_boxes, landmarks = detect_faces_model.predict(img_test_dnet)

        image = show_bboxes(img_test_dnet, bounding_boxes, landmarks)
        image.save("./test/image_landmark.jpg")

        predict_mask = np.squeeze(predict_mask, axis=1)  # [batch, 1, h, w] -> [batch, h, w]

        image = Image.open('./test/image_crop.jpg')

        a = landmarks[0]
        b = a[0:5]
        c = a[5:]
        d = []
        for i, j in zip(b, c):
            d.append([i, j])
        d = np.array(d)

        new_img, new_landmark = Alignment_1(np.array(image), d)
        arr = (np.array(new_img) * 1).astype(np.uint8)
        img = Image.fromarray(arr, 'RGB')
        img.save('./test/my_rgb_image.jpg')

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")

        # =================================================================================


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
        for filename in ['image_crop.jpg', 'image_landmark.jpg', 'image_result.jpg', 'image_mask.jpg', 'my_rgb_image.jpg']:
            zipf.write(os.path.join('test', filename))

    return send_file('images.zip', mimetype='zip')

@app.route('/register_encoder', methods=['POST'])
def register_encoder():
    from flask import request,jsonify
    from werkzeug.utils import secure_filename
    import os
    UPLOAD_FOLDER = 'test'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    path = 'test/register.png'
    image = Image.open(path)
    face_encoder = face_model.encoder(image)
    dict1 = {"face_encoder": face_encoder}
   
    return jsonify(dict1)


@app.route('/verify', methods=['POST'])
def verify():
    from flask import request
    from werkzeug.utils import secure_filename
    import os

    UPLOAD_FOLDER = 'test'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    path = 'test/register.png'
    image = Image.open(path)
    face_encoder = face_model.encoder(image)
    image_root = np.array([[-0.07514341175556183, 0.11126799881458282, -0.12887105345726013, -0.02131621353328228, -0.09985238313674927, -0.21015878021717072, -0.13661301136016846, -0.04636935144662857, 0.12589526176452637, 0.0559636615216732, 0.0056813000701367855, -0.10518907010555267, 0.12293758988380432, 0.13250966370105743, 0.0382891483604908, -0.012248361483216286, 0.03548676148056984, 0.060190923511981964, 0.020137377083301544, 0.34211915731430054, -0.05174599215388298, 0.021402431651949883, 0.030235832557082176, 0.04459521174430847, 0.14984987676143646, 0.05941096693277359, -0.026498526334762573, 0.05537649244070053, -0.03706228360533714, -0.05629302188754082, 0.043990299105644226, -0.014195073395967484, 0.07708245515823364, -0.23202338814735413, -0.17991623282432556, 0.00028674252098426223, -0.11204136162996292, 0.02484668232500553, 0.08071816712617874, -0.16786228120326996, 0.010112942196428776, 0.07408568263053894, 0.09068639576435089, -0.005889218766242266, 0.022411057725548744, 0.09970064461231232, -0.15063799917697906, 0.09050833433866501, 0.016866235062479973, -0.16122522950172424, -0.04303065314888954, 0.16234903037548065, 0.15492117404937744, 0.0024126432836055756, 0.12059826403856277, 0.05830683559179306, -0.04420825093984604, 0.029317006468772888, 0.09487312287092209, 0.10344146937131882, 0.1215548887848854, -0.00800477247685194, 0.1391133964061737, 0.022990528494119644, -0.13601626455783844, 0.03274760767817497, 0.01945088803768158, 0.01277949195355177, 0.00627936003729701, -0.05011270195245743, -0.031712666153907776, 0.04472987726330757, -0.08089784532785416, -0.007073383778333664, -0.013416524976491928, 0.031636402010917664, -0.04235686734318733, -0.0025341592263430357, -0.013831811025738716, -0.030653173103928566, 0.06741253286600113, 0.04134710878133774, -0.07875262200832367, 0.05524411052465439, -0.09251714497804642, 0.010518467985093594, 0.03148442506790161, 0.15430307388305664, 0.08309940993785858, 0.028657587245106697, 0.026017015799880028, 0.06405332684516907, 0.0020244987681508064, 0.04635483771562576, -0.21630172431468964, -0.11544100940227509, -0.0008107014582492411, 0.07089760899543762, 0.016577860340476036, -0.11255726218223572, -0.07742983102798462, -0.03989751264452934, -0.024168146774172783, -0.03353976458311081, 0.012332121841609478, -0.0865689218044281, -0.08406709134578705, -0.024984145537018776, -0.017003636807203293, -0.018863128498196602, 0.003022506134584546, 0.02913896180689335, 0.10429403930902481, 0.132644921541214, 0.053794991225004196, -0.07525547593832016, 0.01891142502427101, 0.16253937780857086, -0.036816611886024475, -0.1574835181236267, -0.012359377928078175, -0.09199711680412292, 0.030821917578577995, 0.0008965338347479701, -0.025790050625801086, 0.004895849619060755, -0.08623595535755157, -0.006098291836678982]])
    face_encoder = np.array(face_encoder)
    probability = np.linalg.norm(image_root - face_encoder, axis=1)
    print(probability)
    if probability < 3.3:
        return "t"
    return "f"


@app.route('/checkin_server', methods=['POST'])
def checkin():
    from flask import request,jsonify
    from werkzeug.utils import secure_filename
    import os
    from zipfile import ZipFile

    UPLOAD_FOLDER = 'test'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    zip_path = 'test/face.zip'
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('test/')

    jsonPath = 'test/json.json'
    with open(jsonPath, 'r') as f:
        data = json.load(f)

    dict_face_oush_distant = {}
    for i in range(len(data)):
        image_in_sqlserver = torch.tensor(json.loads(data[i][1]))
        image_1 = Image.open('./test/dataphoto/captured_image.png')

        probability = face_model.caculate_distant(image_1, image_in_sqlserver)
        dict_face_oush_distant[data[i][0]] = float(probability)

    student_tuplelist_sorted = sorted(dict_face_oush_distant.items(),
                                      key=lambda x: x[1], reverse=True)
    print(student_tuplelist_sorted)
    if student_tuplelist_sorted[-1][1] <0.99:
        dict1 = {"name": student_tuplelist_sorted[-1][0]}
        return jsonify(dict1)
    dict1 = {"name": "未查到数据"}
    return jsonify(dict1)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8900,debug=True)
