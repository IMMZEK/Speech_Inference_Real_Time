import serial
import time

# Open serial port (Replace 'COM5' with whatever port your Arduino is on)
ser = serial.Serial('COM5', 57600)

try:
    while True:
        # Send a string to the Arduino
        ser.write(b"Hello Arduino!\n")
        
        # Check if there is data waiting in the serial buffer
        if ser.in_waiting > 0:
            # Read data out of the buffer until a carriage return/newline is found
            serial_data = ser.readline()
            # Decode bytes to string
            decoded_data = serial_data.decode('utf-8').rstrip()
            # Print the received data
            print("PYReceived:", decoded_data)

        time.sleep(1)  # Sleep for a second
except KeyboardInterrupt:
    ser.close()  # Close serial port when done
