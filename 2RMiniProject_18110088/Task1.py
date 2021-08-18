from render import Renderer
import numpy as np
import cv2

class Robot(Renderer):
    def __init__(self, recordLocation = None):
        super().__init__(recordLocation=recordLocation)
        self.q1=0
        self.q2=0
        self.l1=300
        self.l2=200
        self.t=0

        self.omega=np.pi/6

        self.trace=[]
        self.count=0

    def step(self, dt):
        self.t+=dt

        x=200+100*(np.cos(self.t*self.omega))
        y=-200+100*(np.sin(self.t*self.omega))

        theta = (x**2+y**2-self.l1**2-self.l2**2)/(2*self.l1*self.l2)
        theta = np.arccos(theta)

        self.q1 =np.arctan(y/x)-np.arctan(self.l2*np.sin(theta)/(self.l1+self.l2*np.cos(theta)))

        self.q2 = theta + self.q1

       
        
    def getInfo(self):
        info = {
            "Position" : self.pos
        }
        return info

    def draw(self, image):
       

        j1 = (int(300+self.l1 * np.cos(self.q1-np.pi/4)) , int(500+self.l1 * np.sin(self.q1-np.pi/4)))

        j2 = (int(j1[0] + self.l2 * np.cos(self.q2-np.pi/4)) , int(j1[1] + self.l2 * np.sin(self.q2-np.pi/4)))

        self.pos=j2

        if(self.count%5==0):
            self.trace.append(self.pos)
        self.count+=1

        cv2.line(image,(300,500), j1, (0,255,0), 2)
        cv2.line(image,j1, j2, (0,0,255), 2)
        cv2.circle(image,j2,10,(255,0,0),-1)

        for i in self.trace:
            cv2.circle(image,i,5,(2,5,10),-1)

        return image


robot = Robot(recordLocation = "Task1_animation2.mp4")


for i in range(500):
    robot.step(0.1)
    robot.render()
    