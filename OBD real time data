import obd
import time

#connection = obd.OBD("COM5") # create connection with USB 0
connection = obd.OBD("/dev/ttyUSB0")
while True:
    cmd = obd.commands.SPEED # select an OBD command (sensor)
    rbm = connection.query(cmd) # send the command, and parse the response
    print(rbm.value) # returns unit-bearing values thanks to Pint    
    
    print("-"*20)
    
    
    cmd = obd.commands.RPM 
    response = connection.query(cmd)  
    print(response.value) 
    
    print("-"*20)
    
    cmd = obd.commands.ENGINE_LOAD 
    response = connection.query(cmd)  
    print(response.value) 
    
    print("-"*20)
    
    cmd = obd.commands.COOLANT_TEMP 
    response = connection.query(cmd)  
    print(response.value) 
    
    print("-"*20)
    
    cmd = obd.commands.DISTANCE_W_MIL 
    response = connection.query(cmd)  
    print(response.value) 
    
    print("-"*20)
    time.sleep(10)
