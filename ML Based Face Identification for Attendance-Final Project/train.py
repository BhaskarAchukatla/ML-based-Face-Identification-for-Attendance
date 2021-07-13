#Libraries_Used

#For Opening File Open Dialog 
from tkinter import filedialog 

#Google Text To Speech
import gtts 

#For Playing Voice Clip
from playsound import playsound 

#For Creating Graphical User Interface   
from tkinter import *
import tkinter as tk
from tkinter import Message ,Text

#CV2 is Nothing OpenCV library which is used to Capture Images 
import cv2,os
from cv2 import cv2

#It is used copying, moving, or removing files
import shutil

#Used to enter data into CSV files
import csv

#Numpy is nothing but Multi Dimentional Arrays in Python
import numpy as np

#PIL(Pillow) is used for opening, manipulating, and saving images
from PIL import Image, ImageTk

#For data manipulation and analysis
import pandas as pd

import datetime
import time

import tkinter.ttk as ttk
import tkinter.font as font

#End of Libraries



window = tk.Tk()
window.title("Face_Recogniser")

 
#window.geometry('1280x720')
window.configure(background='black')

#window.attributes('-fullscreen', True)

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

#path = "profile.jpg"

#Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
#img = ImageTk.PhotoImage(Image.open(path))

#The Label widget is a standard Tkinter widget used to display a text or image on the screen.
#panel = tk.Label(window, image = img)


#panel.pack(side = "left", fill = "y", expand = "no")

#cv_img = cv2.imread("img541.jpg")
#x, y, no_channels = cv_img.shape
#canvas = tk.Canvas(window, width = x, height =y)
#canvas.pack(side="left")
#photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img)) 
# Add a PhotoImage to the Canvas
#canvas.create_image(0, 0, image=photo, anchor=tk.NW)

#msg = Message(window, text='Hello, world!')

# Font is a tuple of (font_family, size_in_points, style_modifier_string)



message = tk.Label(window, text="   ML based Face Identification For Attendance" ,bg="blue"  ,fg="white"  ,width=48  ,height=3,font=('times', 30, 'italic bold ')) 

message.place(x=200, y=20)
###################################
lbl = tk.Label(window, text="Enter ID",width=20  ,height=2  ,fg="red"  ,bg="yellow" ,font=('times', 15, ' bold ') ) 
lbl.place(x=400, y=200)
###################################
txt = tk.Entry(window,width=20  ,bg="white" ,fg="red",font=('times', 15, ' bold '))
txt.place(x=700, y=215) #119
###################################

lbl2 = tk.Label(window, text="Enter Name",width=20  ,fg="red"  ,bg="yellow"    ,height=2 ,font=('times', 15, ' bold ')) 
lbl2.place(x=400, y=300)

txt2 = tk.Entry(window,width=20  ,bg="white"  ,fg="red",font=('times', 15, ' bold ')  )
txt2.place(x=700, y=315)

lbl3 = tk.Label(window, text="Notification : ",width=20  ,fg="red"  ,bg="yellow"  ,height=2 ,font=('times', 15, ' bold underline ')) 
lbl3.place(x=400, y=400)

message = tk.Label(window, text="" ,bg="white"  ,fg="green"  ,width=30  ,height=2,font=('times', 15, ' bold ')) 
message.place(x=700, y=400) #153, #175

lbl3 = tk.Label(window, text="Attendance Status: ",width=20  ,fg="red"  ,bg="yellow"  ,height=2 ,font=('times', 15, ' bold  underline')) 
lbl3.place(x=400, y=650)


message2 = tk.Label(window, text="" ,fg="green"   ,bg="white",width=30  ,height=3  ,font=('times', 15, ' bold ')) 
message2.place(x=700, y=650)

#message2 = tk.Label(window, text="" ,fg="green"   ,bg="white",activeforeground = "green",width=30  ,height=3  ,font=('times', 15, ' bold ')) 
#message2.place(x=700, y=650)
 
def clear():  #271
    txt.delete(0, 'end')    
    res = ""
    message.configure(text= res)

def clear2():
    txt2.delete(0, 'end')    
    res = ""
    message.configure(text= res)    
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
 
def TakeImages():        
    Id=(txt.get()) 
    name=(txt2.get())
    if(is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0 #for taking n no of images
        while(True):
            ret, img = cam.read() #ret stores true or false values ..if true cam is opened
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces: #drawing rectangle box 
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                #incrementing sample number 
                sampleNum=sampleNum+1
                #saving the captured faces in TrainingImage folder
                cv2.imwrite("TrainingImage\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                #display the frame
                cv2.imshow('frame',img)
            #wait for 100 miliseconds 
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 10
            elif sampleNum>59:
                break
        cam.release()
        cv2.destroyAllWindows() 
        res = "Images Saved for ID : " + Id +" Name : "+ name #153
        row = [Id , name]
        #all student details are stored in StudentDetails folder with ID and Name
        with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text= res) #82
    else:
        if(is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text= res)
        if(name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text= res)
    
    #End of TakeImages
    
def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    #recognizer = cv2.face_LBPHFaceRecognizer.create()#recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
    #File for Human Face detection developed by OpenCv
    #harcascadePath = "D:\Final Project\haarcascade_frontalface_default.xml"
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath) #To classify faces
    faces,Id = getImagesAndLabels("TrainingImage") #182 #function call
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    #recognizer.save("TrainingImageLabel\STrainner.yml")
    res = "Images are Trained"#+",".join(str(f) for f in Id)
    message.configure(text= res) #82

def getImagesAndLabels(path):
    #To get the path of all images of student which is in Training Image Folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    #imagePaths=[loc1,loc2...] 
    #print(imagePaths)
    
    #create empty face list
    faces=[]   #faces=[array1,array2...]
    #create empty ID list
    Ids=[]     #Ids=[601,602...]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image 
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image file name
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids #170



def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)    
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   
            if(conf < 50):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
                
            else:
                Id='Unknown'                
                tt=str(Id)  
            #if(conf > 75):
                #noOfFile=len(os.listdir("ImagesUnknown"))+1
                #It will store all unkown images
                #cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('q')):
            break
    
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    ########### (counting no of presenties)
    total=str(len(attendance.index))
    ###########
    res=attendance

    message2.configure(text= res)

    
    #Voice Clip will say number of Presenties

    if int(total)==0:
        t1 = gtts.gTTS("No Face is detected")  
        t1.save("zero.mp3") 
        playsound("zero.mp3")
        os.remove("zero.mp3")
    elif int(total)==1:
        t1 = gtts.gTTS(total+" student is present")  
        t1.save("one.mp3") 
        playsound("one.mp3")
        os.remove("one.mp3")
    elif int(total)>1:
        t1 = gtts.gTTS(total+" students are present")  
        t1.save("more.mp3") 
        playsound("more.mp3")
        os.remove("more.mp3")



def PhotoAttendance():
    recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("StudentDetails\StudentDetails.csv")
    
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)    
    #####
    #ret, im =cam.read()
    #photo=face_recognition.load_image_file('example.jpg')

    #window.filename=filedialog.askopenfilename(initialdir="D:\BhaskarAchukatla\Pictures\Camera Roll",title="Select a Photo",filetypes=(("jpg files","*.jpg"),("all files","*.*")))
    window.filename=filedialog.askopenfilename(initialdir="",title="Select a Photo",filetypes=(("jpg files","*.jpg"),("all files","*.*")))

    location=window.filename

    photo=cv2.imread(location)

    gray=cv2.cvtColor(photo,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)    
    for(x,y,w,h) in faces:
        #cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   
        if(conf < 50):
            ts = time.time()      
            date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            aa=df.loc[df['Id'] == Id]['Name'].values
            tt=str(Id)+"-"+aa
            attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
                
        else:
            Id='Unknown'                
            tt=str(Id)  
        #if(conf > 75):
            #noOfFile=len(os.listdir("ImagesUnknown"))+1
            #It will store all unkown images
            #cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
        #cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
    attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
    #cv2.imshow('im',im)


    
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    
    #cv2.destroyAllWindows()
    #print(attendance)
    total=str(len(attendance.index))
    res=attendance
    ###Pygame

    ###
    message2.configure(text= res)
    if int(total)==0:
        t1 = gtts.gTTS("No Face is detected")  
        t1.save("zero.mp3") 
        playsound("zero.mp3")
        os.remove("zero.mp3")
    elif int(total)==1:
        t1 = gtts.gTTS(total+" student is present")  
        t1.save("one.mp3") 
        playsound("one.mp3")
        os.remove("one.mp3")
    elif int(total)>1:
        t1 = gtts.gTTS(total+" students are present")  
        t1.save("more.mp3") 
        playsound("more.mp3")
        os.remove("more.mp3")


#
img1 = ImageTk.PhotoImage(Image.open("D:\python programs\images\LOGO_136x132.jpg")) 
img_label=Label(image=img1)
img_label.place(x=230,y=25)
#



                                                      #92
clearButton = tk.Button(window, text="Clear", command=clear  ,fg="white"  ,bg="red"  ,width=10  ,height=1 ,activebackground = "Black" ,font=('times', 15, ' bold '))
clearButton.place(x=950, y=205)
clearButton2 = tk.Button(window, text="Clear", command=clear2  ,fg="white"  ,bg="red"  ,width=10  ,height=1, activebackground = "Red" ,font=('times', 15, ' bold '))
clearButton2.place(x=950, y=305)  


#Uploading Photo

upload_photo = tk.Button(window, text="Photo Attendance", command=PhotoAttendance  ,fg="white"  ,bg="green"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
upload_photo.place(x=1100, y=400)

                                                        #122  
takeImg = tk.Button(window, text="Provide Image Samples", command=TakeImages  ,fg="white"  ,bg="blue"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
takeImg.place(x=200, y=500)
#Y is row
#X is column                                                                                                           bg color when we click
trainImg = tk.Button(window, text="Train Images", command=TrainImages  ,fg="red"  ,bg="yellow"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
trainImg.place(x=500, y=500)
trackImg = tk.Button(window, text="Live Attendace", command=TrackImages  ,fg="white"  ,bg="green"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
trackImg.place(x=800, y=500)
quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg="white"  ,bg="red"  ,width=20  ,height=3, activebackground = "black" ,font=('times', 15, ' bold '))
quitWindow.place(x=1100, y=500)
copyWrite = tk.Text(window, background=window.cget("background"), borderwidth=0,font=('times', 25, 'italic bold '))
copyWrite.tag_configure("superscript", offset=10)
copyWrite.insert("insert", "Developed by Bhaskar, Pravalika, Jayakrishna, Priyadharshini", "superscript")
copyWrite.configure(state="disabled",fg="red"  )
copyWrite.pack(side="left")
copyWrite.place(x=600, y=750)
 
window.mainloop()

