import serial as ser
se = ser.Serial("/dev/ttyTHS1", 9600, timeout=1)
while 1:
    se.write(b'\x01')

