from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import serial
import threading

app = Flask(__name__)
socketio = SocketIO(app)
ser = serial.Serial('COM5', 57600)

def read_from_port():
    while True:
        data = ser.readline().decode().strip()
        if data:
            socketio.emit('newdata', {'data': data}, namespace='/test')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    t = threading.Thread(target=read_from_port)
    t.start()
    socketio.run(app, debug=True)
