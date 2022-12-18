from datetime import *
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import os
def sparte():
    print("*"*40)

# path ="static\image\pic-2.png"
url ="..\media\photos\Catgroy\\"


# print(cv2.__version__)

class Image:
    def __init__(self,img_name):
        self.img_name=img_name
    
    def showImage(self,img):
        cv2.imshow("my first iamge",img)

    def closedImage(self,img):
        k=cv2.waitKey(0)
        if k==27:
            cv2.destroyAllWindows()
        elif ord("s"):
            cv2.imwrite(f"{url}this a new image.png",img)
            cv2.destroyAllWindows()
                    
    def getPath(self):
        path=f"{url}{self.img_name}"
        # print(path)

        return path
        
    def drawOnImage(self,img):
        
        # imgAfter=cv2.line(img,(0,0),(125,125),(250,255,0),10)
        # imgAfter=cv2.arrowedLine(img,(0,0),(125,125),(0,255,0),10)
        # imgAfter=cv2.rectangle(img,(125,125),(255,255),(0,255,0,),10)
        # imgAfter=cv2.circle(img,(255,255),86,(0,255,0,),10) # add -1 ites be fill
        # font=cv2.FONT_HERSHEY_SIMPLEX
        # imgAfter=cv2.putText(img,"Open cv",(244,244),font,1,(0,255,255),5)
        # imgAfter=cv2.ellipse()
        pts =np.array([[30,20],[40,30],[80,20],[50,10]],np.int32)
        imgAfter=cv2.polylines(img,[pts],True,(255,255,0))
        return imgAfter

    def readImage(self,index=1):
        #call read function
        if index==1:
            img=cv2.imread(self.getPath(),-1)
        else:
            img=self.img_name
            
        img= self.drawOnImage(img)
        #call show function
        self.showImage(img)
        # call closed function
        self.closedImage(img)



class Video:
    def showVideo(self):
        # start a voido by camara or call by path
        cap=cv2.VideoCapture(0)
        #prepare the void and remove the noise and choise here the 'XVID'
        fource=cv2.VideoWriter_fourcc(*"XVID")
        output=cv2.VideoWriter(f"{url}output.avi",fource,20.0,(640,480))

        # print(output.isOpened())
        # print(cap.isOpened())
        # print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        while cap.isOpened():
            rat,frame=cap.read()
            if rat==True:
                
                output.write(frame)
                
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                cv2.imshow("frame",gray)
                cv2.imshow("frame",gray)
                if cv2.waitKey(1) & 0xFF ==ord("q"):
                    break
            else:
                break
            
        cap.release()
        output.release()
        cv2.destroyAllWindows()




# first  : is number of matrix
# seconde: is number of Row 's matrix
# third  : is number of Column 's matrix



# img=np.ones([512,512,3],np.uint8)
# img=np.zeros([3,4,5],dtype=int)

# print(img)

# iamge=Image(img)
# iamge.readImage(index=2)



# imgName="cart-item-1.png"
# iamge=Image(imgName)
# iamge.readImage()

# imageBase=ImageBase()
# imageBase.iamgeBase(img)




# video=Video()
# video.showVideo()










def summer():

    img=np.zeros((600,900,3),dtype=np.uint8)

    cv2.rectangle(img,(0,0),(900,500),(250,225,85),-1)
    cv2.rectangle(img,(0,500),(900,600),(75,180,70),-1)

    cv2.circle(img,(400,150),60,(0,225,255),-1)
    cv2.circle(img,(400,150),70,(220,225,255),5)

    cv2.line(img,(600,500),(600,420),(30,65,155),25)

    plts=np.array([[500,440],[700,400],[600,75]],dtype=np.int32)
    cv2.fillPoly(img,[plts],(75,200,70))
    
    font=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    cv2.putText(img,"the Sun",(700,100),font,1,(255,0,0),5)

    cv2.imshow("my first iamge",img)
    k=cv2.waitKey(0)

    if k==27:
        cv2.destroyAllWindows()
    elif k==ord("s"):
        cv2.imwrite(f"{url}new.png",img)
        cv2.destroyAllWindows()
    
# summer()





def testVoideo():
    # start a voido by camara or call by path
    cap=cv2.VideoCapture(0)
    #prepare the void and remove the noise and choise here the 'XVID'
    fource=cv2.VideoWriter_fourcc(*"XVID")
    output=cv2.VideoWriter(f"{url}output.avi",fource,20.0,(640,480))

    # print(output.isOpened())
    # print(cap.isOpened())
    # print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # cap.set(3,600)
    # cap.set(4,600)
    # sparte()
    # print(cap.get(3))
    # print(cap.get(4))

    while cap.isOpened():
        rat,frame=cap.read()
        if rat==True:
            
            font=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
            textFrame=f"width : {cap.get(3)} , hight : {cap.get(4)}"
            dateFrame=str(datetime.now())
            frame=cv2.putText(frame,textFrame,(40,200),font,1,(200,250,0),3)
            frame=cv2.putText(frame,dateFrame,(40,230),font,1,(200,250,0),3)
            output.write(frame)
            
            # gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            cv2.imshow("frame",frame)
            # cv2.imshow("frame",gray)
            if cv2.waitKey(1) & 0xFF ==ord("q"):
                break
        else:
            break
        

            
    cap.release()
    output.release()
    cv2.destroyAllWindows()


# events=[i for i in dir(cv2) if 'EVENT' in i]
# print(events)

def click_event(event,x,y,flags,param):
    global img
    if event==cv2.EVENT_LBUTTONDOWN:
        print(x,y)
        font=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        text=f"{str(x)} , {str(x)} "
        img=cv2.putText(img,text,(x,y),font,.5,(200,250,0),2)
        cv2.imshow("image", img)
        
    if event==cv2.EVENT_RBUTTONDOWN:
        blue=img[x,y,0]
        green=img[x,y,1]
        red=img[x,y,2]
        
        font=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        text=f"{str(blue)} , {str(green)} , {str(red)}"

        img=cv2.putText(img,text,(x,y),font,.5,(200,250,0),2)
        cv2.imshow("image", img)


def click_event_create_window(event,x,y,flags,param):
    if event==cv2.EVENT_RBUTTONDOWN:
        blue=img[x,y,0]
        green=img[x,y,1]
        red=img[x,y,2]
        myColorsImage=np.zeros((512,512,3),dtype=np.uint8)
        myColorsImage[:]=[blue,green,red]
        cv2.imshow("Color", myColorsImage)


def click_event_draw(event,x,y,flags,param):
    points=[]
    global img
    if event ==cv2.EVENT_LBUTTONDOWN:
        points.append((x,y))
        if len(points)>2:
            cv2.line(img,points[-1],points[-2],(255,0,255),3)
        # if  len(points)>10:
        #     font=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        #     img=cv2.putText(img,"finsh",(x,y),font,.5,(200,250,0),2)
        cv2.imshow("image",img)
        



def click_event_photoshope(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
         print((x,y))


# img=np.zeros((600,900,3),dtype=np.uint8)
# imgName="cat-04.jpg"
# imgName2="cart-item-1.png"

# img=f"{url}{imgName}"
# img2=f"{url}{imgName2}"

# img=cv2.imread(img)
# img2=cv2.imread(img2,-1)

# print(img.shape)
# print(img.size)
# print(img2.size)

# img=cv2.resize(img,(400,500))

# rook=img[0: 100, 0: 100 ]
# print(rook.shape)
# rook=cv2.resize(rook,(600,384 ))
# print(rook.shape)
# print(img.shape)

# img[200: 300 , 200 :300 ]=rook


# [b,g,r]=cv2.split(img)
# print(b,g,r)
# img=cv2.merge((b,g,r))

# dst=cv2.addWeighted(img,.2,rook,.8,0)

#-----------------------------------------------

# cv2.imshow("image",dst)
# cv2.imshow("image2",img2)

# # cv2.setMouseCallback('image',click_event)
# cv2.setMouseCallback('image',click_event_draw)
# cv2.setMouseCallback('image',click_event_create_window)

# cv2.setMouseCallback('image',click_event_photoshope)

# img1=np.zeros((250,512,3),dtype=np.uint8)
# cv2.rectangle(img1,(125,125),(200,200),(255,255,255),2)

# img2=np.full((250,512,3),(123,23,220),dtype=np.uint8)

# img2=np.full((250,512,3),(250),dtype=np.uint8)
# cv2.rectangle(img2,(125,125),(200,200),(0,2,0),2)

# bit_and=cv2.bitwise_and(img1,img2)

# cv2.imshow("imge",img)
# cv2.imshow("imge2",img2)
# cv2.imshow("imge3",bit_and)


# k=cv2.waitKey(0)
# if k==27:
#     cv2.destroyAllWindows()
# elif k==ord("s"):
#     cv2.imwrite(f"{url}new.png",img)
#     cv2.destroyAllWindows()

#-----------------------------------------------


def noThing(x):
    # print(x)
    pass




def trackbarColor():
    # create image black color by 0
    img=np.zeros((300,512,3),dtype=np.uint8)
    # create windows name image
    cv2.namedWindow("image")

    # set track : name of track and name of window
    cv2.createTrackbar("B","image",0,255,noThing)
    cv2.createTrackbar("G","image",0,255,noThing)
    cv2.createTrackbar("R","image",0,255,noThing)

    switch="O OFF\n1: ON"
    cv2.createTrackbar(switch,"image",0,1,noThing)

    while(1):
        # show image im new windows its'name image
        cv2.imshow("image",img)
        k=cv2.waitKey(1)
        # key from key borads 
        if k ==27:
            break
        # get track value by the name of track and name of window
        b=cv2.getTrackbarPos("B","image")
        g=cv2.getTrackbarPos("G","image")
        r=cv2.getTrackbarPos("R","image")
        s=cv2.getTrackbarPos(switch,"image")
        # if its no its no work , or on its work 
        if s==0:
            img[:]=0 # make the color is black
        else:
            img[:]=[b,g,r] # indifer the volor by valus b , g ,r
    # remove screen image
    cv2.destroyAllWindows()




# trackbarColor()

#----------------------------------------------
imgName="cat-04.jpg"


def rgbToHSV():
    img=f"{url}{imgName}"
    frame=cv2.imread(img,0)
    while 1:
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        l_p=np.array([110,50,50])
        u_p=np.array([130,255,255])
        
        mask=cv2.inRange(hsv,l_p,u_p)
        
        res=cv2.bitwise_and(frame,frame,mask=mask)

        cv2.imshow("frame",frame)
        cv2.imshow("mask",mask)
        cv2.imshow("res",res)
        
        k=cv2.waitKey(0)
        # key from key borads 
        if k ==27:
            break
            
        cv2.destroyAllWindows()


# rgbToHSV()


# img=f"{url}{imgName}"




# img=cv2.imread(img,0)


# _,thr1=cv2.threshold(img,55,255,cv2.THRESH_BINARY)
# _,thr2=cv2.threshold(img,200,255,cv2.THRESH_BINARY_INV)
# _,thr3=cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
# _,thr4=cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
# _,thr5=cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
# thr6=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
# thr7=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

# cv2.imshow("thr1",thr1)
# cv2.imshow("thr2",thr2)
# cv2.imshow("thr3",thr3)
# cv2.imshow("thr4",thr4)
# cv2.imshow("thr5",thr5)
# cv2.imshow("thr6",thr7)

# k=cv2.waitKey(0)
# if k==27:
#     cv2.destroyAllWindows()

# plt.imshow(img)



#------------------------------------------------------------

# _,mask=cv2.threshold(img,220,255,cv2.THRESH_BINARY_INV)

# kernal=np.ones((5,5),np.uint8)

# dilation=cv2.dilate(mask,kernal,iterations=2)#good but increass the balls size because if any pixels under the kernal
# erode=cv2.erode(mask,kernal,iterations=2)#it encode the balls because if all pixels under the karnal not one erod
# open=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernal,iterations=2)
# close=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernal,iterations=2)



# cv2.imshow("img",img)
# cv2.imshow("mask",mask)
# cv2.imshow("dilation",dilation)
# cv2.imshow("erode",erode)
# cv2.imshow("open",open)
# cv2.imshow("close",close)

# k=cv2.waitKey(0)
# if k==27:
#     cv2.destroyAllWindows()



#------------------------------------------------------------


# blur=cv2.blur(img,(5,5))#mean for pixels
# gblur=cv2.GaussianBlur(img,(5,5),0)#for noise image
# mgblur=cv2.medianBlur(img,5)#for salt and paper
# bilateralFilter=cv2.bilateralFilter(img,9,75,75)#preserve the borders


# cv2.imshow("img",img)
# cv2.imshow("blur",blur)
# cv2.imshow("gblur",gblur)
# cv2.imshow("mgblur",mgblur)
# cv2.imshow("bilateralFilter",bilateralFilter)


# k=cv2.waitKey(0)
# if k==27:
#     cv2.destroyAllWindows()

#------------------------------------------------------------

# lap=cv2.Laplacian(img,cv2.CV_64F,ksize=3)
# lap=np.uint8(np.absolute(lap))

# cv2.imshow("img",img)
# cv2.imshow("lap",lap)


# k=cv2.waitKey(0)
# if k==27:
#     cv2.destroyAllWindows()






















#AHMED IBRAME SAEED
#--------------------------------------------
pathImg="C:\Projects\\traingPython\Ecommerce website\greatkart_template\greatkart\images"

def AhmedSaedd():
    # img=np.zeros((250,512,3),dtype=np.uint8)
    os.chdir(f"{pathImg}/items")
    #read images
    img="2.jpg"
    img=cv2.imread(img,0)
    
    #apply some thing
    
    # lap=cv2.Laplacian(img,cv2.CV_64F,ksize=3)
    # lap=np.uint8(np.absolute(lap))
    
    # sobel_x=cv2.Sobel(img,cv2.CV_64F,1,0)
    # sobel_y=cv2.Sobel(img,cv2.CV_64F,0,1)
    
    # sobel_x=np.uint8(np.absolute(sobel_x))
    # sobel_y=np.uint8(np.absolute(sobel_y))
    
    # compine_sobel_x_y=cv2.bitwise_or(sobel_x,sobel_y)
    # cany=cv2.Canny(img,100,100)
    
    # lr1=cv2.pyrDown(img)#minmize the image
    # lr2=cv2.pyrDown(lr1)
    # lr3=cv2.pyrUp(lr1)#mixmize image
    
  
    
    #show images
    cv2.imshow("my first iamge",img)
    # cv2.imshow("lap",lap)
    # cv2.imshow("sobel_x",sobel_x)
    # cv2.imshow("sobel_y",sobel_y)
    # cv2.imshow("compine_sobel_x_y",compine_sobel_x_y)
    # cv2.imshow("cany",cany)
    
    # cv2.imshow("lr1",lr1)
    # cv2.imshow("lr2",lr2)
    # cv2.imshow("lr3",lr3)
    
    
    layer=img.copy()
    gp=[layer]
    for i in range(6):
        layer=cv2.pyrDown(layer)
        
        # layer=cv2.Laplacian(layer,cv2.CV_64F,ksize=3)
        
        gp.append(layer)
        cv2.imshow(str(i),layer)
        
        
    #close windows
    k=cv2.waitKey(0)
    if k==27:
        cv2.destroyAllWindows()

# AhmedSaedd()

def ImageBlending():
    os.chdir(f"{pathImg}/items")
    img1="apple 1.jpg"
    img2="apple 2.jpg"
    
   #read images
   
    img1=cv2.imread(img1,-1)
    img2=cv2.imread(img2,-1)
    
    #resize image
    #percent by which the image is resized
    scale_percent = 30

    #calculate the 50 percent of original dimensions
    width = int(img1.shape[1] * scale_percent / 100)
    height = int(img1.shape[0] * scale_percent / 100)
    #dsize
    dsize = (width, height)
    img1 = cv2.resize(img1, dsize)
    img2 = cv2.resize(img2, dsize)
    #
    
    # print(img1.shape)
    # print(img2.shape)
    
    
    #mearge two image
    
    #cut the image
    # img1_img2=np.hstack((img1[:,235:523],img2[:,200:488]))
    
    # print(img1_img2)
    #genrate gaussian pyramids for img1
    img1_copy=img1.copy()
    ga_img1=[img1_copy]
    for i in range(6):
        img1_copy=cv2.pyrDown(img1_copy)
        ga_img1.append(img1_copy)
     
    #genrate laplacian pyramids for img1
    img1_copy=ga_img1[5]
    lap_img1=[img1_copy]
    for i in range(5,0,-1):
        gaussian_expanded=cv2.pyrUp(ga_img1[i])
        lapacian=cv2.subtract(ga_img1[i],gaussian_expanded)
        lap_img1.append(lapacian)
         

    #genrate gaussian pyramids for img2
    img2_copy=img2.copy()
    ga_img2=[img2_copy]
    for i in range(6):
        img2_copy=cv2.pyrDown(img2_copy)
        ga_img2.append(img2_copy)
        
    #genrate laplacian pyramids for img2
    img2_copy=ga_img2[5]
    lap_img2=[img2_copy]
    for i in range(5,0,-1):
        gaussian_expanded=cv2.pyrUp(ga_img2[i])
        lapacian=cv2.subtract(ga_img2[i-1],gaussian_expanded)
        lap_img2.append(lapacian)
     
    #now add left image and right image
    imag1_imag2_pyramids=[]
    
    for lap1,lap2 in zip(lap_img1,lap_img2):
        cols,row,sh=lap1.shape
        lapacian=np.hstack((lap1[:,0:int(cols/2)],lap2[:,int(cols/2):]))
        imag1_imag2_pyramids.append(lapacian)
    
    #now reconstruct
    img1_img2_reconstruct=imag1_imag2_pyramids[0]
    cv2.imshow("img 0",img1_img2_reconstruct)
    for i in range(1,6):
        img1_img2_reconstruct=cv2.pyrUp(img1_img2_reconstruct)
        img1_img2_reconstruct=cv2.add(imag1_imag2_pyramids[i],img1_img2_reconstruct)
    
    #show image
    cv2.imshow("img 1",img1)
    cv2.imshow("img 2",img2)
    cv2.imshow("img1_img2_reconstruct",img1_img2_reconstruct)
    # cv2.imshow("img1_img2",img1_img2)
    
    #close image 
    k=cv2.waitKey(0)
    if k==27:
        cv2.destroyAllWindows()
        
        
# ImageBlending()


def contours():
    #read image
    os.chdir(f"{pathImg}/misc")
    img="playmarket.png"
    img=cv2.imread(img)
    img=cv2.resize(img,(400,200))

    #
    imgray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    rat,threshold=cv2.threshold(imgray,20,255,0)
    contours,hierarchy=cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    print(len(contours))
    print(contours[0])
    
    cv2.drawContours(img,contours,-1,(0,255,0),1)
    #
    #show image
    cv2.imshow("img",img)
    cv2.imshow("img gray",imgray)
    cv2.imshow("img threshold",threshold)
    # cv2.imshow("img contours",contours)
    #colse image
    k=cv2.waitKey(0)
    if k==27:
        cv2.destroyAllWindows()
        

# contours()
def dffdf():
    videoPath="C:\\Users\gamal\Videos\my video"
    os.chdir(videoPath)
    videoImage="WIN_20210701_10_41_58_Pro.mp4"
    cap=cv2.VideoCapture(0)
    
    
    fource=cv2.VideoWriter_fourcc(*"XVID")
    output=cv2.VideoWriter(f"{videoPath}output.avi",fource,20.0,(640,480))
    
    # print(output.isOpened())
    # print(cap.isOpened())
    
    rat,frame1=cap.read()
    rat,frame2=cap.read()
    
    # ,int(cap.get(3))
    # int(cap.get(4))
    
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while cap.isOpened():
        diff=cv2.absdiff(frame1,frame2)
        gray=cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
        
        blur=cv2.GaussianBlur(gray,(5,5),0)
        
        _,threshold=cv2.threshold(blur,60,255,cv2.THRESH_BINARY)
        
        dilated=cv2.dilate(threshold,None,iterations=10)
        
        contourst,hierarchy=cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        
        for contour in contourst:
            (x,y,w,h)=cv2.boundingRect(contour)
            
            if cv2.contourArea(contour)<900:
                continue
            
            cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame1,"status; {}".format("Movement"),(10,20),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3
                        )
        image=cv2.resize(frame1,(1280,720))
        output.write(image)
            
        #
        cv2.imshow("frame",frame1)
        
        frame1=frame2
        rat,frame2=cap.read()
        
        if cv2.waitKey(60)==27:
            break
        
    cv2.destroyAllWindows()
    cap.release()
    output.release()




basePath="C:\Projects\Collage\ML\computer vision"
baseData=f"{basePath}\\video"
# nameVideo="Egg Counter _ Camera-based Eggs Counting for Conveyor Belts with Computer Vision & AI.mp4"
# nameVideo="Egg counter with computer vision .mp4"
nameVideo="cars counting.mp4"
#veh


def page_center(x,y,w,h):
    h=int(h/2)
    w=int(w/2)
    cx=x+h
    cy=y+w
    return cx,cy


def ObjectDetectionCounting(nameVideo, y1=300):
    os.chdir(baseData)
    
    cap=cv2.VideoCapture(nameVideo)
    # cap.set(3,100)
    # cap.set(4,100)
    # BGS=cv2.createBackgroundSubtractorMOG2()
    # BGS=cv2.createBackgroundSubtractorKNN()
    
    # BGS = cv2.bgsegm.createBackgroundSubtractorMOG() 
    
    BGS= cv2.createBackgroundSubtractorMOG2()
    
    ww=80
    hh=80
    
    detec=[]
    offset=6
    count=0
    while cap.isOpened():
        #read fram 
        rat,fram1=cap.read()
        #1128.0 700.0
        # fram1=fram1[80:700, 80:700]
        # print(cap.get(3),cap.get(4))
        
        # fram1 = cv2.rotate(fram1, cv2.ROTATE_90_CLOCKWISE)
        # fram1=cv2.resize(fram1,(int(cap.get(3)/1.5),int(cap.get(4)/1.5)))
        
        if fram1 is None:
            break
        #transform into gray  
        gray=cv2.cvtColor(fram1,cv2.COLOR_BGR2GRAY)
        #blur
        blur=cv2.GaussianBlur(gray,(3,3),5)
        #subtrationg
        img_sub=BGS.apply(blur)
        
        dilat=cv2.dilate(img_sub,np.ones((5,5)))
        
        contour,hierarchy=cv2.findContours(dilat,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            
        cv2.line(fram1,(25,y1),(1200,y1),(255,0,0),3)
        
        for (i,c) in enumerate(contour):
            (x,y,w,h)=cv2.boundingRect(c)
            
            valider_contorno=(w>=ww)and (h>=hh)
            
            if not valider_contorno:
                continue
            
            cv2.rectangle(fram1,(x,x),(x+w,y+h),(0,255,0),2)
            center=page_center(x, y, w, h)
        
            detec.append(center)
        
            cv2.circle(fram1,center,4,(0,0,255),-1)
            
            for (x,y) in detec:
                if y<(y1+offset) and y>(y1-offset):
                    count+=1
                    cv2.line(fram1,(25,y1),(1200,y1),(172,88,3),3)
                    detec.remove((x,y))
                    print(f"No. cars{count}")
                    
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(fram1,f"Nm. Object is {count}",(25,y1),font,1,(244,45,2),4)
    
    
        cv2.imshow("frame",fram1)
        # cv2.imshow("gray",gray)
        # cv2.imshow("blur",blur)
        cv2.imshow("img_sub",img_sub)
        
        # key=cv2.waitKey(30)
        if cv2.waitKey(1) & 0xFF ==ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()


# ObjectDetectionCounting(0,y1=80)


def putTextOnImage(img,text,x,y):
    font=cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(img,text,(x,y),font,0.5,(3,6,4))
    return img

# img=np.ones([512,512,3],np.uint8)
# img.fill(255)
def imagedrawingby():
    baseData=f"{basePath}\\image"
    os.chdir(baseData)
    img="Untitled.png"
    img=cv2.imread(img,-1)
    img=cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2),))
    
    # print((int(img.shape[1]/2),int(img.shape[0]/2),))
    
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,thresh=cv2.threshold(gray,240,255,cv2.CHAIN_APPROX_NONE)
    countours,_=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                    
    # dat=pd.DataFrame(countours).reshape()
    # print(dat)
    
    for contour in countours:
        # (x,y,w,h)=cv2.boundingRect(contour)
        # cv2.rectangle(img,(x,y),((x+w),(y+h)),(244,34,55),4)
        approx=cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
        # print(approx)
        
        cv2.drawContours(img,[approx],0,(180,3,8),5)
        (x,y,w,h)=cv2.boundingRect(approx)
        aspectRato=w/h
        x=approx.ravel()[0]
        y=approx.ravel()[1]
        if len(approx)==3:
            putTextOnImage(img,"Triangle",x,y)
        elif len(approx)==4:
            if aspectRato>=0.95 and aspectRato<=1.5:
                putTextOnImage(img,"squre",x,y)
            else:
                putTextOnImage(img,"Rectangle",x,y)
    
        elif len(approx)==10:
            putTextOnImage(img,"Star",x,y)
        elif len(approx)==5:
            putTextOnImage(img,"Pentagon",x,y)
        else:
            putTextOnImage(img,"Circle",x,y)
            
    cv2.imshow('image',img)
    # cv2.imshow('gray',gray)
    # cv2.imshow('threshold',thresh)
    
    
    k=cv2.waitKey(0)
    
    if k==27:
        cv2.destroyAllWindows()
    elif k==ord("s"):
        cv2.imwrite(f"{url}new.png",img)
        cv2.destroyAllWindows()


    
    
# imagedrawingby()


# import cv2 
# import face_recognition
# import numpy as np
# import csv
# import os
# from datetime import datetime

# video_capture = cv2.VideoCapture(0)
# jobs_image = face_recognition.load_image_file("photos/jobs.jpg")
# jobs_encoding = face_recognition. face_encoding(jobs_image)[0]

# ratan_tata_image = face_recognition.load_image_file("photos/tata.jpg")
# ratan_tata_encoding= face_recognition. face_encoding(ratan_tata_image)[0]

# sadmona_image = face_recognition.load_image_file("photos/sadmona.jpg")
# sadmona_encoding = face_recognition. face_encoding(sadmona_image)[0]
 
# tesla_image = face_recognition.load_image_file("photos/tesla.jpg")
# tesla_encoding = face_recognition. face_encoding(tesla_image)[0]

# known_face_encoding = [
#     jobs_encoding,
#     ratan_tata_encoding,
#     sadmona_encoding,
#     tesla_encoding
#     ]
# known_faces_names =[
# "jobs",
# "ratan tata",
# "sadmona",
# "tesla"
# ]

# students known_faces_names.copy()





def deawingImageByPython():
    
    pathImage=f"{basePath}\image\gamal.jpg"
    
    from sketchpy import canvas
    from sklearn.preprocessing import scale
    
    # pic=canvas.sketch_from_svg()
    
    # pic.draw()
    
    
    image=cv2.imread(pathImage,-1)
    image=cv2.resize(image,(470,430))
    gray_iamge=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    invert=cv2.bitwise_not(gray_iamge)
    
    blur=cv2.GaussianBlur(invert,(21,21),0)
    
    invertedblur=cv2.bitwise_not(blur)
    sketch=cv2.divide(gray_iamge,invertedblur,scale=256.0)
    
    # cv2.imwrite("testing.png",sketch)
    
    
    cv2.imshow('image',sketch)
    
    k=cv2.waitKey(0)
    
    if k==27:
        cv2.destroyAllWindows()
    elif k==ord("s"):
        cv2.imwrite(f"{url}new.png",img)
        cv2.destroyAllWindows()



def close(img):
    k=cv2.waitKey(0)

    if k==27:
        cv2.destroyAllWindows()
    elif k==ord("s"):
        cv2.imwrite(f"{url}new.png",img)
        cv2.destroyAllWindows()


def Histogramscontraststretching():    
    
    pathImage=f"{basePath}\image\gamal.jpg"
    
    # img=np.ones((500,500),np.uint8)
    img=cv2.imread(pathImage)
    size=(400,300)
    img=cv2.resize(img,size)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    b,g,r=cv2.split(img)
    
    # print(img.ravel())
    # plt.hist(img.ravel(),255,[0,255])
    
    # plt.hist(b.ravel(),255,[0,255])
    # plt.hist(g.ravel(),255,[0,255])
    # plt.hist(r.ravel(),255,[0,255])
    
    #or 
    
    hist=cv2.calcHist(img,[0],None,[255],[0,250])
    plt.plot(hist)
    
    
    plt.show()
    cv2.imshow('image',img)
    
    #spliting images
    
    cv2.imshow('r',r)
    cv2.imshow('g',g)
    cv2.imshow('b',b)
    
    
    close(img)
    

# Histogramscontraststretching()



def OTS():#NOT COMPLETE 
    
    pathImage=f"{basePath}\image\\test.png"
    img=cv2.imread(pathImage)
    size=(500,350)
    img=cv2.resize(img,size,0)
    
    ret1,thr1=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    
    # ret2,thr2=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    blur=cv2.GaussianBlur(img,(5,5),0)
    
    # ret3,thr3=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    imags=[img,0,thr1,
       img,0,thr2,
       blur,0,thr3]
    
    
    
    cv2.imshow("img",img)
    
    close(img)


def Objectdetecing():
    pathImage=f"{basePath}\image\gamal.jpg"
    
    # img=np.ones((500,500),np.uint8)
    img=cv2.imread(pathImage)
    # img=cv2.GaussianBlur(img,(5,5),0)
    shape=img.shape
    size=(int(shape[0]*0.7),int(shape[1]*0.4))
    img=cv2.resize(img,size)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    
    face=gray[10:370,170:350]
    #       hight   width
    w,h=face.shape[::-1]
    
    # print("Wigth ",w," hight ",h)
    
    res=cv2.matchTemplate(gray,face,cv2.TM_CCORR_NORMED)
    # print(res)
    
    threshold=0.99
    loc=np.where(res>=threshold)
    
    # print(loc)
    font=cv2.FONT_HERSHEY_COMPLEX
    
    for pt in zip(*loc[::-1]): #zip(loc[1],loc[0])
        print(pt)
        # cv2.putText(img,"gamal",(pt),font,1,(255,209,45),0)
        cv2.rectangle(img,(pt),(pt[0]+w,pt[1]+h),(0,255,0),2)
    
    cv2.imshow('image',img)
    # cv2.setMouseCallback('image',click_event)
    
    cv2.imshow('face',face)
    cv2.imshow('gray',gray)
    cv2.imshow('res',res)
    close(img)
    
# Objectdetecing()
# face_cascade=cv2.

def faceDetection():
    pathImage=f"{basePath}\image\gamal.jpg"
    
    img=cv2.imread(pathImage)
    shape=img.shape
    img=cv2.resize(img,(int(shape[0]*0.6),int(shape[1]*0.4)))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    cv2.imshow("img",img)
    close(img)
    
    
    
def EyesDetections():    
    pathVideo="C:\\Users\gamal\OneDrive - Faculty of Computers and Information Technology (Zagazig University)\Pictures\Camera Roll"
    os.chdir(pathVideo)
    cap=cv2.VideoCapture("gamal.mp4")
    g=cv2.COLOR_BGR2GRAY
    while True:
        rat,frame=cap.read()
        if rat==False:
            break
        frame=cv2.resize(frame,(500,500))
        roi=frame[220:300,220:350]#eyes
        #gray on frame and roi
        # gray=cv2.cvtColor(frame,g)
        gray_roi=cv2.cvtColor(roi,g)
        gray_roi=cv2.GaussianBlur(gray_roi,(7,7),0)
        _,col,row=roi.shape[::-1]
        
        
        _,threshold=cv2.threshold(gray_roi,75,255,cv2.THRESH_BINARY_INV)
        # threshold=cv2.resize(threshold,(500,500))
        contoursy,_=cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        contoursf=sorted(contoursy,key=lambda x:cv2.contourArea(x),reverse=True)
        
        for con in contoursf:
            # print(con)
            x,y,w,h=cv2.boundingRect(con)
            cv2.rectangle(roi,(x,y),(x+w,y+h),(25,234,0),1)
            cv2.line(roi,(x+int(w/2),0),(x+int(w/2),row),(34,234,0),2)
            cv2.line(roi,(0,y+int(h/2)),(col,y+int(h/2)),(25,200,0),2)
            # print(x,y,w,h)
            break
        # res=cv2.matchTemplate(gray_roi,gray,cv2.TM_CCORR_NORMED)
        
        # threshold=0.999
        # loc=np.where(res>=threshold)
        
        
        # for pt in zip(*loc[::-1]): #zip(loc[1],loc[0])
        #     cv2.rectangle(frame,(pt),(pt[0]+w,pt[1]+h),(0,255,0),1)
    
        
        cv2.imshow("frame",frame)
        cv2.imshow("threshold",threshold)
        # cv2.imshow("gray_roi",gray_roi)
        cv2.imshow("roi",roi)
        if cv2.waitKey(1) & 0xFF ==ord("q"):
            break
        
    cap.release()
    cv2.destroyAllWindows()


# EyesDetections()

#---------------> 41 <-------------------

def faceLandMarks():
    # import dlib
    
    # print(dlib.__version__)
    
    detector=dlib.get_frontal_face_detecor()
    predictor=dlib.shape_predictor("shape_predicor_68_face_landmarks.dat")
    
    cap=cv2.VideoCapture(0)
    while True:
        ret,frame=cap.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=detector(gray)
        for face in faces:
            x1=face.left()
            y1=face.top()
            x2=face.righs()
            y2=face.bottom()
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
            landmarks=predictor(gray,face)
            for n in range(0,68):
                x=landmarks.part(n).x
                y=landmarks.part(n).y
                cv2.circle(frame,(x,y),3,(0,255,0),-1)
                
        cv2.imshow("frame",frame)
        if cv2.waitKey(1) & 0xFF ==ord("q"):
            break
            
    cap.release()
    cv2.destroyAllWindows()


#---------------> 42 <-------------------

#


def GranhamScan():
    pathVideo="C:\\Users\gamal\OneDrive - Faculty of Computers and Information Technology (Zagazig University)\Pictures\Camera Roll"
    os.chdir(pathVideo)
    cap=cv2.VideoCapture(0)
    g=cv2.COLOR_BGR2GRAY
    while True:
        rat,frame=cap.read()
        if rat==False:
            break
        # frame=cv2.imread("image\gamal.jpg")
        frame=cv2.resize(frame,(500,500))
        gray=cv2.cvtColor(frame,g)
        gray=cv2.GaussianBlur(gray,(7,7),0)
    
        ret,threshold=cv2.threshold(gray,55,255,0)
        # # threshold=cv2.resize(threshold,(500,500))
        contoursy,_=cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        bfilter=cv2.bilateralFilte(gray,11,17,17)
        
        # for i in range(len(contoursy)):
        #     # print(con)
        #     hull=cv2.convexHull(contoursy[i])
        #     cv2.drawContours(frame,[hull],-1,(33,244,0),2)
        
        cv2.imshow("frame",frame)
        cv2.imshow("threshold",threshold)
        cv2.imshow("threshold",bfilter)
        # close(frame)
        if cv2.waitKey(1) & 0xFF ==ord("q"):
            break
        
    cap.release()
    cv2.destroyAllWindows()

  
# GranhamScan()
# --------------44-------------





os.chdir(f"{basePath}/image")
g=cv2.COLOR_BGR2GRAY
frame=cv2.imread("gamal.jpg")
frame=cv2.resize(frame,(500,500))
gray=cv2.cvtColor(frame,g)
# gray=cv2.GaussianBlur(gray,(7,7),0)

ret,threshold=cv2.threshold(gray,105,255,0)
# # threshold=cv2.resize(threshold,(500,500))
contoursy,_=cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
bfilter=cv2.bilateralFilter(gray,25,17,17)

# for i in range(len(contoursy)):
#     # print(con)
#     hull=cv2.convexHull(contoursy[i])
#     cv2.drawContours(frame,[hull],-1,(33,244,0),2)

# cv2.imshow("frame",frame)
# cv2.imshow("threshold",threshold)
cv2.imshow("bfilter",bfilter)
close(frame)




