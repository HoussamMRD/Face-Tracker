   ###################################################    CONFIG-CODE    #################################################################

import cv2
import os
from flask import Flask, request, render_template , redirect, url_for
from datetime import date, datetime ,  timedelta    # Import datetime module
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask import jsonify
import pymysql
pymysql.install_as_MySQLdb()



app = Flask(__name__)

# Configure the static directory
app.static_folder = 'static'

app.config['SECRET_KEY'] = 'houssamMRD007'
app.config['UPLOAD_FOLDER'] = 'static/files'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:houssamMRD007@localhost:3306/ai'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Initialize Flask-Migrate
migrate = Migrate(app, db)


###################################################    CLASS-CODE    #################################################################

class Employe(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    profession = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(100), nullable=False)
    address = db.Column(db.String(100), nullable=False)
    dateNaissance = db.Column(db.Date, nullable=False)
    dateEmbauche = db.Column(db.Date, nullable=False)



@app.route('/employes_list', methods=['GET'])
def employes_list():
    search_query = request.args.get('search')
    if search_query:
        employes = Employe.query.filter(Employe.name.ilike(f'%{search_query}%')).all()
    else:
        employes = Employe.query.all()
    return render_template('Emp/employes_list.html', employes=employes)


@app.route('/employes_Add', methods=['GET', 'POST'])
def employes_Add():
    if request.method == 'POST':
        name = request.form.get('name')
        profession = request.form.get('profession')
        email = request.form.get('email')
        phone = request.form.get('phone')
        address = request.form.get('address')

        # Convert date strings to Python date objects
        dateNaissance = datetime.strptime(request.form.get('dateNaissance'), '%Y-%m-%d').date()
        dateEmbauche = datetime.strptime(request.form.get('dateEmbauche'), '%Y-%m-%d').date()


        employe = Employe(name=name, profession=profession, email=email, phone=phone, address=address,
                          dateNaissance=dateNaissance, dateEmbauche=dateEmbauche)
        db.session.add(employe)
        db.session.commit()

        return redirect(url_for('employes_list'))

    return render_template('Emp/employes_Add.html')

@app.route('/employes_Edit/<int:id>', methods=['GET', 'POST'])
def employes_Edit(id):
    employe = Employe.query.get_or_404(id)
    if request.method == 'POST':
        employe.name = request.form.get('name')
        employe.profession = request.form.get('profession')
        employe.email = request.form.get('email')
        employe.phone = request.form.get('phone')
        employe.address = request.form.get('address')

        # Convert date strings to Python date objects
        employe.dateNaissance = datetime.strptime(request.form.get('dateNaissance'), '%Y-%m-%d').date()
        employe.dateEmbauche = datetime.strptime(request.form.get('dateEmbauche'), '%Y-%m-%d').date()


        db.session.commit()
        return redirect(url_for('employes_list'))

    return render_template('Emp/employes_Edit.html', employe=employe)


@app.route('/employes_Delete/<int:id>', methods=['GET', 'POST'])
def employes_Delete(id):
    employe = Employe.query.get_or_404(id)
    db.session.delete(employe)
    db.session.commit()
    return redirect(url_for('employes_list'))

@app.route('/ViewEmp/<int:id>', methods=['GET', 'POST'])
def ViewEmp(id):
    employe = Employe.query.get_or_404(id)
    return render_template('Emp/ViewEmp.html', employe=employe)

######################################################################

class Camera(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    type = db.Column(db.String(100), nullable=False)
    ref = db.Column(db.String(100),  nullable=False)
    department = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return f"<Camera {self.name}>"



@app.route('/CameraList')
def CameraList():
    search_query = request.args.get('search')
    if search_query:
        cameras = Camera.query.filter(Camera.name.ilike(f'%{search_query}%')).all()
    else:
      cameras = Camera.query.all()
      print(cameras)
    return render_template('Cameras/CameraList.html', cameras=cameras)



@app.route('/CameraAdd', methods=['GET', 'POST'])
def CameraAdd():
    if request.method == 'POST':
        name = request.form['name']
        type = request.form['type']
        ref = request.form['ref']
        department = request.form['department']

        new_camera = Camera(name=name, type=type, ref=ref, department=department)
        db.session.add(new_camera)
        db.session.commit()


        return redirect(url_for('CameraList'))

    return render_template('Cameras/CameraAdd.html')





@app.route('/CameraEdit/<int:id>', methods=['GET', 'POST'])
def CameraEdit(id):
    camera = Camera.query.get_or_404(id)
    if request.method == 'POST':
        camera.name = request.form['name']
        camera.type = request.form['type']
        camera.ref = request.form['ref']
        camera.department = request.form['department']

        db.session.commit()
        return redirect(url_for('CameraList'))

    return render_template('Cameras/CameraEdit.html', camera=camera)


@app.route('/CameraDelete/<int:id>', methods=['GET', 'POST'])
def CameraDelete(id):
    camera = Camera.query.get_or_404(id)
    db.session.delete(camera)
    db.session.commit()
    return redirect(url_for('CameraList'))








###################################################    AI-CODE    #################################################################

nimgs = 10

imgBackground = cv2.imread("background.png")

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d/%B/%Y")


face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')


def totalreg():
    return len(os.listdir('static/faces'))


def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))

        # Apply Non-Maximum Suppression
        face_points = non_max_suppression(face_points)

        return face_points
    except:
        return []


def non_max_suppression(boxes, overlapThresh=0.5):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")


def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

def preprocess_face(face):
    # Resize the face image to the desired size (e.g., 128x128)
    resized_face = cv2.resize(face, (128, 128))
    # Flatten the image to a 1D array
    flattened_face = resized_face.flatten()
    # Convert to numpy array
    facearray = np.array([flattened_face])
    return facearray


def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


def extract_attendance():
       # Read the CSV file for today's attendance
       df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')

       # Drop duplicate rows based on 'Name', 'Roll', and 'Time' columns
       df = df.drop_duplicates(subset=['Name', 'Roll', 'Time'])

       # Extract data
       names = df['Name'].tolist()
       rolls = df['Roll'].tolist()
       times = df['Time'].tolist()
       l = len(df)

       return names, rolls, times, l


def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if str(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')


def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l



@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()

    # Calculate the number of unique attendees
    unique_attendees = len(set(names))

    current_time = datetime.now().strftime("%H:%M:%S")  # Get current time
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, current_time=current_time , unique_attendees=unique_attendees)  # Pass current time to template







last_recognition_time = datetime.min

@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)
    global last_recognition_time
    detection_cadres = {}  # Dictionary to store detection cadres for each face
    while ret:
        ret, frame = cap.read()
        faces = extract_faces(frame)
        current_time = datetime.now()
        for (x, y, w, h) in faces:
            if (current_time - last_recognition_time).total_seconds() >= 10:  # Keep detection cadre for 10 seconds
                detection_cadres[(x, y, w, h)] = current_time
            if (current_time - detection_cadres.get((x, y, w, h), datetime.min)).total_seconds() < 10:  # Draw detection cadre if within 10 seconds
                cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
                cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
                face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
                identified_person = identify_face(face.reshape(1, -1))[0]
                add_attendance(identified_person)
                last_recognition_time = current_time  # Update the last recognition time
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
                cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
                cv2.putText(frame, f'{identified_person}', (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
        imgBackground[162:162 + 480, 55:55 + 640] = frame
        cv2.imshow('Face-Tracker', imgBackground)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)





@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == nimgs*5:
            break
        cv2.imshow('Adding new Employe', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template('Dash.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)



###################################################    APP-ROUTE   #################################################################




@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username == 'RH-Manager' and password == 'RH1234':
            return redirect(url_for('Dash'))

    return render_template('login.html')




@app.route('/Dash', methods=['GET', 'POST'])
def Dash():
    names, rolls, times, l = extract_attendance()

    # Calculate the number of unique attendees
    unique_attendees = len(set(names))

    Absence_Emp = totalreg() - unique_attendees
    current_capacity = unique_attendees / totalreg() * 100

    current_time = datetime.now().strftime("%H:%M:%S")  # Get current time
    return render_template('Dash.html' , names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, current_time=current_time , unique_attendees=unique_attendees , Absence_Emp=Absence_Emp ,  current_capacity = current_capacity )   # Pass current time to template




@app.route('/attendance', methods=['GET'])
def attendance():
    names, rolls, times, l = extract_attendance()
    return render_template('Attendance/attendance.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/Add-Emp', methods=['GET', 'POST'])
def AddEmp():
    return render_template('Attendance/Add-Emp.html')



@app.route('/Archive', methods=['GET'])
def archive():
    # This route will just render the HTML page
    return render_template('Attendance/Archive.html')

@app.route('/fetch-attendance', methods=['GET'])
def fetch_attendance():
    date_requested = request.args.get('date', None)
    if date_requested:
        try:
            date_obj = datetime.strptime(date_requested, '%Y-%m-%d')
            # Assuming the CSV filename includes the date in 'mm_dd_yy' format
            df = pd.read_csv(f'Attendance/Attendance-{date_obj.strftime("%m_%d_%y")}.csv')
        except FileNotFoundError:
            return jsonify([])  # Return an empty list if no file is found for the date
        records = df.to_dict(orient='records')
        return jsonify(records)
    else:
        return jsonify([])



@app.route('/edit-attendance/<int:id>', methods=['GET', 'POST'])
def edit_attendance(id):
    record = Attendance.query.get_or_404(id)
    if request.method == 'POST':
        record.name = request.form['name']
        record.roll = request.form['roll']
        record.time = request.form['time']  # Make sure to parse datetime
        db.session.commit()
        return redirect(url_for('show_attendance'))
    return render_template('edit_attendance.html', record=record)








@app.route('/delete-attendance/<int:id>', methods=['DELETE'])
def delete_attendance(id):
    record = Attendance.query.get_or_404(id)  # Adjust this to your actual model and DB query

    try:
        db.session.delete(record)
        db.session.commit()
        return jsonify({'success': True})
    except:
        db.session.rollback()
        return jsonify({'success': False})

















if __name__ == '__main__':
    app.run(debug=True)

###################################################    END-CODE    #################################################################