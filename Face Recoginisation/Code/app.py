import check_camera
import capture_image
import train_image
import recognize
import os
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
@app.route('/home')
def home():
    return render_template("index.html")


# check_camera function
@app.route('/Check_camera', methods=["post", "GET"])
def Check_camera():
    check_camera.camer()
    return render_template("index.html")


# add_student function data from add_student side
@app.route('/add_student_data', methods=["post", "GET"])
def add_student_data():
    return render_template("add_student.html")


# add_student function
@app.route('/add_student', methods=["post", "GET"])
def add_student():
    output = request.form.to_dict()
    name = output["name"]
    roll = output["Roll number"]
    print(roll)
    print(name)
    capture_image.takeImages_interface(roll, name)
    return render_template("index.html")


# train function
@app.route('/Train', methods=["post", "GET"])
def Train():
    name = train_image.ti()
    return render_template("index.html", name=name)


# Give_attendance function
@app.route('/Give_attendance', methods=["post", "GET"])
def Give_attendance():
    recognize.recognize_attendence()
    return render_template("index.html")


# Attendance function
@app.route('/Attendance', methods=["post", "GET"])
def Attendance():
    os.startfile(r'.\Attendance')
    return render_template("index.html")


# Quit Function
@app.route('/Quit', methods=["post", "GET"])
def Quit():
    exit()


if __name__ == '__main__':
    app.run(debug=True, port=5001)

# add_student()
