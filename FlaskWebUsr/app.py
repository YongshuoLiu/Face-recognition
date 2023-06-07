from flask import Flask, render_template, request, send_from_directory, make_response,redirect,url_for
import os
import base64
import cv2
import numpy as np
import requests
from zipfile import ZipFile
from datetime import datetime
import time
from flask import jsonify


app = Flask(__name__)


@app.route('/checkinform')
def checkinform():
    return render_template('checkform.html')


@app.route('/show')
def show_form():
    return render_template('index.html')


@app.route('/register')
def register():
    # 返回register.html页面
    return render_template('register.html')


@app.route('/verify_from')
def verify_from():
    return render_template('verify.html')


@app.route('/verify', methods=['POST'])
def verify():

    image_b64 = request.form.get('imageBase64')
    image_data = base64.b64decode(image_b64.split(',')[1])
    image_np = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_np, flags=1)
    cv2.imwrite(os.path.join('dataphoto', 'register.png'), image)
    print("capture success")
    url = 'http://10.8.255.226:30315/verify'
    file_path = 'dataphoto/register.png'  # replace with your actual file path

    with open(file_path, 'rb') as file:
        response = requests.post(url, files={'file': file})

        if response.text == "t":
            # return "http://127.0.0.1:5000/register"
            redirect_url = url_for('register')  # 设置重定向 URL
            return jsonify({'redirect_url': redirect_url})

    redirect_url = url_for('verify_from')  # 设置重定向 URL
    return jsonify({'redirect_url': redirect_url})


@app.route('/resister_start', methods=['POST'])
def resister_start():
    import json
    time_start = time.time()  # 开始计时
    fname = request.form.get('fname')
    image_b64 = request.form.get('imageBase64')
    # image_b64 = request.values['imageBase64']
    image_data = base64.b64decode(image_b64.split(',')[1])
    image_np = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_np, flags=1)
    cv2.imwrite(os.path.join('dataphoto', 'register.png'), image)
    print("capture success")
    url = 'http://10.8.255.226:30315/register_encoder'
    file_path = 'dataphoto/register.png'  # replace with your actual file path
    with open(file_path, 'rb') as file:
        response = requests.post(url, files={'file': file})
        face_coder = response.json()['face_encoder']
    print(response.json())
    print(face_coder)
    dict1 = {"name":fname,"coder": face_coder}

    json_data = json.dumps(dict1)
    requests.post('http://localhost:8900/upload_sql',json=json_data)

    time_end = time.time()  # 结束计时
    time_c = time_end - time_start  # 运行所花时间
    print('注册时间消耗', time_c, 's')
    return ""

# 签到
@app.route('/checkin', methods=['POST'])
def checkin():
    import json
    time_start = time.time()  # 开始计时
    image_b64 = request.values['imageBase64']
    image_data = base64.b64decode(image_b64.split(',')[1])
    image_np = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_np, flags=1)
    cv2.imwrite(os.path.join('dataphoto', 'captured_image.png'), image)
    print("capture success")

    # Send a single image to the server
    url = 'http://10.8.255.226:30315/checkin_server'
    facetxt = requests.post('http://localhost:8900/get_face')

    with open('json.json', 'w', encoding='utf-8') as file:
        file.write(json.dumps(facetxt.json(), indent=2, ensure_ascii=False))
    # Create a ZipFile object

    with ZipFile('face.zip', 'w') as zipf:
        for filename in ['dataphoto/captured_image.png', 'json.json']:
            zipf.write(filename)
    name = ""
    with open('face.zip', 'rb') as file:
        response = requests.post(url, files={'file': file})
        name = response.json()["name"]
    now = str(datetime.now())
    print(name,now)
    if name != "未查到数据":

        dict1 = {"name":name,"time": now}
        json_data = json.dumps(dict1)
        requests.post('http://localhost:8900/insert_log',json=json_data)
        results = "签到成功：" + name
        time_end = time.time()  # 结束计时
        time_c = time_end - time_start  # 运行所花时间
        print('签到时间消耗', time_c, 's')
        return results
    time_end = time.time()  # 结束计时
    time_c = time_end - time_start  # 运行所花时间
    print('签到时间消耗', time_c, 's')
    return ""


@app.route('/')
def index():
    return render_template('root.html')

# @app.route('/')
# def index():
#     return make_response(open('index.html').read())

@app.after_request
def add_header(response):
    # 告诉浏览器不要缓存响应
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/capture', methods=['POST'])
def capture():
    image_b64 = request.values['imageBase64']
    image_data = base64.b64decode(image_b64.split(',')[1])
    image_np = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image_np, flags=1)
    cv2.imwrite(os.path.join('dataphoto', 'captured_image.png'), image)

    print("capture success")

    # Send a single image to the server
    url = 'http://10.8.255.226:30315/upload'
    file_path = 'dataphoto/captured_image.png'  # replace with your actual file path
    with open(file_path, 'rb') as file:
        response = requests.post(url, files={'file': file})

    zip_path = os.path.join('dataphoto', 'images.zip')
    with open(zip_path, 'wb') as file:
        file.write(response.content)
    # Unzip the file

    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('dataphoto/')
    return ""



@app.route('/get_image')
def get_image():
    image_result = 'image_result.jpg'
    return send_from_directory('dataphoto/test/', image_result)


@app.route('/get_image2')
def get_image2():

    image_crop = 'image_crop.jpg'

    return send_from_directory('dataphoto/test/', image_crop)


@app.route('/get_image3')
def get_image3():

    image_landmark = 'image_landmark.jpg'

    return send_from_directory('dataphoto/test/', image_landmark)


@app.route('/get_image5')
def get_image5():
    rotate = 'my_rgb_image.jpg'
    return send_from_directory('dataphoto/test/', rotate)


@app.route('/get_image4')
def get_image4():
    myrgb = 'my_rgb_image.jpg'
    return send_from_directory('dataphoto/test/', myrgb)



if __name__ == '__main__':
    app.run(host='127.0.0.1',port=5000,debug=True)
