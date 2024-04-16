import serial
import time

class UARTHelper:
    def __init__(self, port='COM5', baud_rate=57600):
        self.ser = serial.Serial(port, baud_rate)

    def send_data(self, data):
        """ Send data to the microcontroller via UART """
        self.ser.write(data.encode() + b"\n")

    def receive_data(self):
        """ Receive data from the microcontroller via UART """
        if self.ser.in_waiting > 0:
            data = self.ser.readline().decode('utf-8').rstrip()
            return data
        return None

    def close(self):
        """ Close the UART connection """
        self.ser.close()
