from render import Renderer
import numpy as np
import cv2
import scipy
from scipy.integrate import odeint

class Robot(Renderer):
    def __init__(self,x0,y0,x,y,recordLocation = None):
        super().__init__(recordLocation=recordLocation)

        self.l1=300
        self.l2=200
       
        ori=self.cordtoang(x0,y0)
        self.q10=ori[0]
        self.q20=ori[1]

        ini=self.cordtoang(x,y)
        self.q1=ini[0]
        self.q2=ini[1]
        
        self.t1=0
        self.t2=0

        self.fx=1000
        self.fy=800


    def step(self, dt):

        if(abs(self.q10-self.q1)<=0.01 and abs(self.q20-self.q2)<=0.01):
            fx = self.fx
            fy = self.fy

            self.t1 = fy*self.l1*np.cos(self.q1)-fx*self.l1*np.sin(self.q1)
            self.t2 = fy*self.l2*np.cos(self.q2)-fx*self.l2*np.sin(self.q2)
        else:
            self.q1+=(self.q10-self.q1)*dt
            self.q2+=(self.q20-self.q2)*dt
            
    def cordtoang(self,x,y):
        theta = (x**2+y**2-self.l1**2-self.l2**2)/(2*self.l1*self.l2)
        theta = np.arccos(theta)

        q1 =np.arctan(y/x)-np.arctan(self.l2*np.sin(theta)/(self.l1+self.l2*np.cos(theta)))

        q2 = theta + q1

        return (q1,q2)
        
    def getInfo(self):
        info = {
            "Position" : self.pos,
            "Torque" : (-self.t1,-self.t2)
        }
        return info

    def draw(self, image):
       

        j1 = (int(300+self.l1 * np.cos(-self.q1-np.pi/4)) , int(500+self.l1 * np.sin(-self.q1-np.pi/4)))

        j2 = (int(j1[0] + self.l2 * np.cos(-self.q2-np.pi/4)) , int(j1[1] + self.l2 * np.sin(-self.q2-np.pi/4)))

        self.pos=j2

        cv2.line(image,(300,500), j1, (0,255,0), 2)
        cv2.line(image,j1, j2, (0,0,255), 2)
        cv2.circle(image,j2,10,(255,0,0),-1)
        cv2.line(image,(80,100),(80,400),(0,0,0),3)

        return image


robot = Robot(100,400,200,100,recordLocation="Task2_animation.mp4")


for i in range(500):
    robot.step(0.03)
    robot.render()
    