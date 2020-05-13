import os
from flask import Flask, send_from_directory
from flask import render_template, request

import imutils
import cv2

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

model_path = "models/instance_norm/udnie.t7"


def run_model():
    image_path = "static/content.jpg"

    net = cv2.dnn.readNetFromTorch(model_path)

    image = cv2.imread(image_path)
    
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1.0, (w, h),
        (103.939, 116.779, 123.680), swapRB=False, 
        crop=False)
    net.setInput(blob)
    output = net.forward()

    output = output.reshape((3, output.shape[2], 
        output.shape[3]))
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.680
    output /= 255.0
    output = output.transpose(1, 2, 0)

    img = cv2.normalize(src=output, dst=None, alpha=0,
    beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    cv2.imwrite("static/output.jpg", img)
#     cv2.imwrite("images/output/output.jpg", img)

@app.route("/", methods=['GET'])
def home():
    return render_template("home.html")


@app.route("/style", methods=['GET'])
def index():

    if 'path' in request.args:
        global model_path
        model_path = "models/instance_norm/"+str(request.args['path'])

    return render_template("test.html")

@app.route("/upload", methods=['POST', 'GET'])
def upload():
    target = os.path.join(APP_ROOT, 'static/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    # file_name=""
    for file in request.files.getlist("myFile[]"):
        print(file)
        filename = file.filename
        # filename="content.jpg"
        destination = "/".join([target, "content.jpg"])
        print(destination)
        file.save(destination)
#         d2 = os.path.join(APP_ROOT, 'images/content/content.jpg')
#         file.save(d2)
        # file_name=filename

    run_model()

    file_name="output.jpg"
    return render_template("complete.html", image_name=file_name)
    # return send_from_directory("images/output", file_name, as_attachment=True)


@app.route("/upload/<filename>")
def send_image(filename):
    return send_from_directory("images/output", filename)


if __name__ == "__main__":
    app.run(debug=True)
