import serial, sys, time
import threading
from enum import Enum


'''
    Converter class, it contains the functions which generate the message in the correct form. 
'''
class MessageConverter:

    '''
        Contains all key words, which can be used between the Nucleo and Raspberry. For each key words are assigned a difference actions. 
    '''
    class MessageKeys(Enum):
        # start enumaration
        MOVE='MCTL'
        BRAKE='BRAK'
        BEZIER='SPLN'
        PID_VALUE='PIDS'
        PID_ACTIVATE='PIDA'
        SAFETY_BRAKE_ACTIVATE='SFBR'
        DISTANCE_SENSOR_ECHOER='DSPB'
        ENCODER_ECHOER='ENPB'
        # end enumaration



        '''
            @name    keyStr
            @brief   
                It returns the key words for the specified actions. 
            @param [in] f_key           an action defined in the enumaration   

            @retval the key word
            
            Example Usage: 
            @code
                key=MessageKeys.MOVE
                keyWords=MessageKeys.keyStr(key)
            @endcode
        '''
        def keyStr(f_key):
            if not type(f_key)==MessageConverter.MessageKeys:
                return None
            else:
                return f_key.value            



    '''
        @name    MCTL
        @brief   
            It generates a message to control the motor speed and the steering angle.
        @param [in] f_vel       motor PWM signal, or, if PID activated, the reference (in cm/s) 
        @param [in] f_angle     steering servo angle

        @retval the formatted message
        
        Example Usage: str_msg=MessageConverter.MCTL(f_vel,f_angle)
        @code
            
        @endcode
    '''
    def MCTL(f_vel=0.0,f_angle=0.0):
        if type(f_vel==float) and type(f_angle==float):
            return "#MCTL:%.2f"%f_vel+";%.2f"%f_angle+";;\r\n"
        else:
            return ""
    '''
        @name    BRAKE
        @brief   
            It generates a message to act on the brake and apply a steering angle.
        @param [in] f_angle     steering servo angle

        @retval the formatted message
        
        Example Usage: str_msg=MessageConverter.BRAKE(f_angle)
        @code
            
        @endcode
    '''
    def BRAKE(f_angle=0.0):
        if type(f_angle)==float:
            return "#BRAK:%.2f"%f_angle+";;\r\n"
        else:
            return ""
    '''
        @name    SPLN
        @brief   
            It generates a message for the actuators to initiate a movement on a spline trajectory with the given parameters.
        @param [in] A           first coordinate on the curve
        @param [in] B           second coordinate on the curve
        @param [in] C           third coordinate on the curve
        @param [in] D           forth coordinate on the curve
        @param [in] f_dur_sec   movemet duration in seconds
        @param [in] isForward   forward/backward movement

        @retval the formatted message
        
        Example Usage: str_msg=MessageConverter.SPLN(f_A,f_B,f_C,f_D,f_dur_sec,isForward)
        @code
            
        @endcode
    '''
    def SPLN(A,B,C,D,dur_sec=1.0,isForward=True):
        if type(dur_sec)==float and type(isForward)==bool:
            isAllComplex=(type(A)==complex and type(B)==complex and type(C)==complex and type(D)==complex)
            isAllList=((type(A)==list and type(B)==list and type(C)==list and type(D)==list) and (len(A==2) and len(B==2) and len(C==2) and len(D==2)))
            isForward=int(isForward)
            if isAllComplex:
                return '#SPLN:%d;'%isForward+'%.2f;'%A.real+'%.2f;'%A.imag+'%.2f;'%B.real+'%.2f;'%B.imag+'%.2f;'%C.real+'%.2f;'%C.imag+'%.2f;'%D.real+'%.2f;'%D.imag+'%.2f;'%dur_sec+';\r\n'
            elif isAllList:
                return '#SPLN:%d;'%isForward+'%.2f;'%A[0]+'%.2f;'%A[1]+'%.2f;'%B[0]+'%.2f;'%B[1]+'%.2f;'%C[0]+'%.2f;'%C[1]+'%.2f;'%D[0]+'%.2f;'%D[1]+'%.2f;'%dur_sec+';\r\n'
            else:
                return ""
        else:
            return ""
    '''
        @name    PIDS
        @brief   
            It generates a message for setting the PID with the given parameters.
        @param [in] kp          proportional factor
        @param [in] ki          integral factor
        @param [in] kd          derivative factor
        @param [in] tf          filter time constant

        @retval the formatted message
        
        Example Usage: str_msg=MessageConverter.PIDS(kp,ki,kd,tf)
        @code
            
        @endcode
    '''
    def PIDS(kp,ki,kd,tf):
        if type(kp)==float and type(ki)==float and type(kd)==float and type(tf)==float:    
            return "#PIDS:%.5f;"%kp+"%.5f;"%ki+"%.5f;"%kd+"%.5f;"%tf+";\r\n"
        else:
            return ""
    '''
        @name    PIDA
        @brief   
            It generates a message for activating PID usage.
        @param [in] activate    boolean value for activating PID

        @retval the formatted message
        
        Example Usage: str_msg=MessageConverter.PIDA(activate)
        @code
            
        @endcode
    '''
    def PIDA(activate=True):
        l_value=0
        if activate:
            l_value=1
        return '#PIDA:%d;'%l_value+';\r\n'
    '''
        @name    SFBR
        @brief   
            It generates a message for activating the safety brake usage.
        @param [in] activate    boolean value for activating safety brake

        @retval the formatted message
        
        Example Usage: str_msg=MessageConverter.SFBR(activate)
        @code
            
        @endcode
    '''
    def SFBR(activate=True):
        l_value=0
        if activate:
            l_value=1
        return '#SFBR:%d;'%l_value+';\r\n'
    '''
        @name    DSPB
        @brief   
            It generates a message for activating the distance sensor echoer usage.
        @param [in] activate    boolean value for activating distance sensor echoer

        @retval the formatted message
        
        Example Usage: str_msg=MessageConverter.DSPB(activate)
        @code
            
        @endcode
    '''
    def DSPB(activate=True):
        l_value=0
        if activate:
            l_value=1
        return '#DSPB:%d;'%l_value+';\r\n'
    '''
        @name    ENPB
        @brief   
            It generates a message for activating the encoder echoer usage.
        @param [in] activate    boolean value for activating safety brake

        @retval the formatted message
        
        Example Usage: str_msg=MessageConverter.ENPB(activate)
        @code
            
        @endcode
    '''
    def ENPB(activate=True):
        l_value=0
        if activate:
            l_value=1
        return '#ENPB:%d;'%l_value+';\r\n'

'''
    ReadThread class, it contains the functions which read incoming serial communication. 
'''
class ReadThread(threading.Thread):
    '''
        CallbackEvent class, it contains the function for adding callbacks to events. 
    '''
    class CallbackEvent:
        '''
            @name    __init__
            @brief   
                Constructor method for the CallbackEvent class.
            @param [in] self         reference to the current instance of the class
            @param [in] event        event triggering callback call
            @param [in] callbackFunc callback to be called when event takes place

            @retval -
            
            Example Usage: -
            @code
                
            @endcode
        '''
        def __init__(self,event,callbackFunc):
            self.event=event
            self.callbackFunc=callbackFunc
    '''
        @name    __init__
        @brief   
            Constructor method for the ReadThread class.
        @param [in] self          reference to the current instance of the class
        @param [in] f_theadID     theadID
        @param [in] f_serialCon   Serial connection object
        @param [in] f_fileHandler FileHandler object 
        @param [in] f_printOut    boolean value indincatin whether ???

        @retval
        
        Example Usage: -
        @code
            
        @endcode
    '''
    def __init__(self,f_theadID,f_serialCon,f_fileHandler,f_printOut=False):
        threading.Thread.__init__(self)
        self.ThreadID=f_theadID
        self.serialCon=f_serialCon
        self.fileHandler=f_fileHandler
        self.Run=False
        self.buff=""
        self.isResponse=False
        self.printOut=f_printOut
        self.Responses=[]
        self.Waiters={}
    
    '''
        @name    run
        @brief   
            Run method for the ReadThread class.
        @param [in] self          reference to the current instance of the class

        @retval
        
        Example Usage: -
        @code
            
        @endcode
    '''
    def run(self):
        while(self.Run):
            read_chr=self.serialCon.read()
            try:
                read_chr=(read_chr.decode("ascii"))
                if read_chr=='@':
                    self.isResponse=True
                    if len(self.buff)!=0:
                        self.checkWaiters(self.buff)
                    self.buff=""
                elif read_chr=='\r':
                    self.isResponse=False
                    if len(self.buff)!=0:
                        self.checkWaiters(self.buff)
                    self.buff=""
                if self.isResponse:
                    self.buff+=read_chr
                self.fileHandler.write(read_chr)    
                if self.printOut:
                    sys.stdout.write(read_chr)
            except UnicodeDecodeError:
                pass
    '''
        @name    checkWaiters
        @brief   
            Method for checking the waiter functions set the ReadThread class and for setting callback events.
        @param [in] self          reference to the current instance of the class
        @param [in] f_response    response transmitted

        @retval none
        
        Example Usage: -
        @code
            
        @endcode
    '''
    def checkWaiters(self,f_response):
        l_key=f_response[1:5]
        if l_key in self.Waiters:
            l_waiters=self.Waiters[l_key]
            for eventCallback in l_waiters:
                eventCallback.event.set()
                if not eventCallback.callbackFunc==None:
                    eventCallback.callbackFunc(f_response[6:-2])
    '''
        @name    addWaiter
        @brief   
            Method for adding a waiter function for the ReadThread class.
        @param [in] self             reference to the current instance of the class
        @param [in] f_key            message key
        @param [in] f_objEvent       event triggering callback call
        @param [in] callbackFunction callback function

        @retval none
        
        Example Usage: serialHandler.readThread.addWaiter("PIDA",ev1)
        @code
            
        @endcode
    '''
    def addWaiter(self,f_key,f_objEvent,callbackFunction=None):
        l_evc=ReadThread.CallbackEvent(f_objEvent,callbackFunction)
        if f_key in self.Waiters:
            obj_events_a=self.Waiters[f_key]
            obj_events_a.append(l_evc)
        else:
            obj_events_a=[l_evc]
            self.Waiters[f_key]=obj_events_a
    '''
        @name    deleteWaiter
        @brief   
            Method for deleting a waiter function for the ReadThread class.
        @param [in] self             reference to the current instance of the class
        @param [in] f_key            message key
        @param [in] f_objEvent       event triggering callback call

        @retval none
        
        Example Usage: serialHandler.readThread.deleteWaiter("MCTL",ev1)
        @code
            
        @endcode
    '''
    def deleteWaiter(self,f_key,f_objEvent):
        if f_key in self.Waiters:
            l_obj_events_a=self.Waiters[f_key]
            for callbackEventObj in l_obj_events_a:
                if callbackEventObj.event==f_objEvent:
                    l_obj_events_a.remove(callbackEventObj)            
    '''
        @name    stop
        @brief   
            Method for stopping a waiter function for the ReadThread class.
        @param [in] self             reference to the current instance of the class

        @retval none
        
        Example Usage: serialHandler.readThread.stop()
        @code
            
        @endcode
    '''
    def stop(self):
        self.Run=False
    '''
        @name    start
        @brief   
            Method for starting a waiter function for the ReadThread class.
        @param [in] self             reference to the current instance of the class

        @retval none
        
        Example Usage: serialHandler.readThread.start()
        @code
            
        @endcode
    '''
    def start(self):
        self.Run=True
        super(ReadThread,self).start()
    
'''
    FileHandler class, it contains the functions for file handling. 
'''
class FileHandler:
    '''
        @name    __init__
        @brief   
            Constructor method for the FileHandler class.
        @param [in] self          reference to the current instance of the class
        @param [in] f_fileName    file name

        @retval
        
        Example Usage: -
        @code
            
        @endcode
    '''
    def __init__(self,f_fileName):
        self.outFile=open(f_fileName,'w')
        self.lock=threading.Lock()
    '''
        @name    write
        @brief   
            Method for writing into file.
        @param [in] self          reference to the current instance of the class
        @param [in] f_str         string to be written

        @retval none
        
        Example Usage: -
        @code
            
        @endcode
    '''
    def write(self,f_str):
        self.lock.acquire()
        self.outFile.write(f_str)
        self.lock.release()        
    
    '''
        @name    close
        @brief   
            Method for closing file.
        @param [in] self          reference to the current instance of the class

        @retval none
        
        Example Usage: -
        @code
            
        @endcode
    '''
    def close(self):
        self.outFile.close()

'''
    SerialHandler class, it contains the functions for sending commands. 
'''
class SerialHandler:

    '''
        @name    __init__
        @brief   
            Constructor method for the SerialHandler class.
        @param [in] self           reference to the current instance of the class
        @param [in] f_device_File  serial device file name
        @param [in] f_history_file name of the file containing command history

        @retval
        
        Example Usage: -
        @code
            
        @endcode
    '''

    def __init__(self,f_device_File='/dev/ttyACM0',f_history_file='historyFile.txt'):
        self.serialCon=serial.Serial(f_device_File,460800,timeout=1)
        self.historyFile=FileHandler(f_history_file)
        self.readThread=ReadThread(1,self.serialCon,self.historyFile)
        self.lock=threading.Lock()
    '''
        @name    startReadThread
        @brief   
            Function for starting reading thread.
        @param [in] self           reference to the current instance of the class

        @retval none
        
        Example Usage: serialHandler.startReadThread()
        @code
            
        @endcode
    '''
 
    def startReadThread(self):
        self.readThread.start()
    '''
        @name    send
        @brief   
            Function for starting reading thread.
        @param [in] self           reference to the current instance of the class
        @param [in] msg            message to be sent

        @retval none
        
        Example Usage: serialHandler.send(str_msg)
        @code
            
        @endcode
    '''

    def send(self,msg):
        # self.historyFile.write(msg)
        self.lock.acquire()
        self.serialCon.write(msg.encode('ascii'))
        self.lock.release()

    '''
        @name    sendMove
        @brief   
            Function for sending move command.
        @param [in] self        reference to the current instance of the class
        @param [in] f_vel       motor PWM signal, or, if PID activated, the reference (in cm/s) 
        @param [in] f_angle     steering servo angle

        @retval success status, True if no error
        
        Example Usage: sent=serialHandler.sendMove(pwm,20.0)
        @code
            
        @endcode
    '''

    def sendMove(self,f_vel,f_angle):
        str_msg=MessageConverter.MCTL(f_vel,f_angle)
        if str_msg!="":
            self.send(str_msg)
            return True
        else:
            return False
    '''
        @name    sendBrake
        @brief   
            Function for sending brake command.
        @param [in] self        reference to the current instance of the class
        @param [in] f_angle     steering servo angle

        @retval success status, True if no error
        
        Example Usage: sent=serialHandler.sendBrake(20.0)
        @code
            
        @endcode
    '''

    def sendBrake(self,f_angle):
        str_msg=MessageConverter.BRAKE(f_angle)
        if str_msg!="":
            self.send(str_msg)
            return True
        else:
            return False
    '''
        @name    sendBrake
        @brief   
            Function for sending bezier command.
        @param [in] self        reference to the current instance of the class
        @param [in] A           first coordinate on the curve
        @param [in] B           second coordinate on the curve
        @param [in] C           third coordinate on the curve
        @param [in] D           forth coordinate on the curve
        @param [in] f_dur_sec   movemet duration in seconds
        @param [in] isForward   forward/backward movement

        @retval success status, True if no error
        
        Example Usage: sent=serialHandler.sendBezierCurve(0.5051175777578528+0.5051175777578528j,0.7840863128094306+0.22614884270627506j,0.7840863128094306-0.22614884270627506j,0.5051175777578528-0.5051175777578528j,3.0,False)
        @code
            
        @endcode
    '''
    def sendBezierCurve(self,f_A,f_B,f_C,f_D,f_dur_sec,isForward):
        str_msg=MessageConverter.SPLN(f_A,f_B,f_C,f_D,f_dur_sec,isForward)
        if str_msg!="":
            self.send(str_msg)
            return True
        else:
            return False
    '''
        @name    sendPidActivation
        @brief   
            Function for sending PID activation command.
        @param [in] self        reference to the current instance of the class
        @param [in] activate    boolean value for activating PID

        @retval success status, True if no error
        
        Example Usage: sent=serialHandler.sendPidActivation(True)
        @code
            
        @endcode
    '''

    def sendPidActivation(self,activate=True):
        str_msg=MessageConverter.PIDA(activate)
        if str_msg!="":
            self.send(str_msg)
            return True
        else:
            return False
    '''
        @name    sendPidValue
        @brief   
            Function for sending PID parameter setting command.
        @param [in] self        reference to the current instance of the class
        @param [in] kp          proportional factor
        @param [in] ki          integral factor
        @param [in] kd          derivative factor
        @param [in] tf          filter time constant

        @retval success status, True if no error
        
        Example Usage: sent=serialHandler.sendPidValue(kp,ki,kd,tf)
        @code
            
        @endcode
    '''

    def sendPidValue(self,kp,ki,kd,tf):
        str_msg=MessageConverter.PIDS(kp,ki,kd,tf)
        if str_msg!="":
            self.send(str_msg)
            return True
        else:
            return False
    '''
        @name    sendSafetyStopActivation
        @brief   
            Function for sending safety brake activation command.
        @param [in] self        reference to the current instance of the class
        @param [in] activate    boolean value for activating safety brake

        @retval success status, True if no error
        
        Example Usage: sent=serialHandler.sendSafetyStopActivation(True)
        @code
            
        @endcode
    '''

    def sendSafetyStopActivation(self,activate=True):
        str_msg=MessageConverter.SFBR(activate)
        if str_msg!="":
            self.send(str_msg)
            return True
        else:
            return False
    '''
        @name    sendDistanceSensorsPublisher
        @brief   
            Function for sending distance sensor publisher activation command.
        @param [in] self        reference to the current instance of the class
        @param [in] activate    boolean value for activating distance sensor publishar

        @retval success status, True if no error
        
        Example Usage: sent=serialHandler.sendDistanceSensorsPublisher(True)
        @code
            
        @endcode
    '''

    def sendDistanceSensorsPublisher(self,activate=True):
        str_msg=MessageConverter.DSPB(activate)
        if str_msg!="":
            self.send(str_msg)
            return True
        else:
            return False
    '''
        @name    sendEncoderPublisher
        @brief   
            Function for sending encoder publisher activation command.
        @param [in] self        reference to the current instance of the class
        @param [in] activate    boolean value for activating encoder publishar

        @retval success status, True if no error
        
        Example Usage: sent=serialHandler.sendEncoderPublisher(True)
        @code
            
        @endcode
    '''

    def sendEncoderPublisher(self,activate=True):
        str_msg=MessageConverter.ENPB(activate)
        if str_msg!="":
            self.send(str_msg)
            return True
        else:
            return False
    '''
        @name    close
        @brief   
            Function for closing communication.
        @param [in] self        reference to the current instance of the class

        @retval none
        
        Example Usage: sent=serialHandler.close()
        @code
            
        @endcode
    '''

    def close(self):
        self.readThread.stop()
        self.readThread.join()
        self.serialCon.close()
        self.historyFile.close()

# if __name__=="__main__":
#     main()
        
        