from render import Renderer, scaleAndShow
import numpy as np
import cv2
import scipy
from scipy.integrate import odeint

class Robot(Renderer):
    def __init__(self,x0,y0,x,y,recordLocation = None):
        super().__init__(recordLocation = recordLocation)
        self.ode = scipy.integrate.ode(self.func).set_integrator('vode', nsteps = 500 , method = 'bdf')
        
        self.l1=300
        self.l2=200
        self.k=1

        ori=self.cordtoang(x0,y0)
        self.q10=ori[0]
        self.q20=ori[1]

        ini=self.cordtoang(x,y)
        self.q1=ini[0]
        self.q2=ini[1]
        self.q1dot=0
        self.q2dot=0

        #moments of inertia
        self.I1=1000
        self.I2=800

    def cordtoang(self,x,y):
        theta = (x**2+y**2-self.l1**2-self.l2**2)/(2*self.l1*self.l2)
        theta = np.arccos(theta)

        q1 =np.arctan(y/x)-np.arctan(self.l2*np.sin(theta)/(self.l1+self.l2*np.cos(theta)))

        q2 = theta + q1

        return (q1,q2)

        

    def func(self,t,y):
        q1=y[0]
        q2=y[1]

        q1dot = y[2]
        q2dot = y[3]

        fx = self.k*(self.l1*np.cos(q1) + self.l2*np.cos(q2) - self.l1*np.cos(self.q10) - self.l2*np.cos(self.q20))
        fy = self.k*(self.l1*np.sin(q1) + self.l2*np.sin(q2) - self.l1*np.sin(self.q10) - self.l2*np.sin(self.q20))
        
        # Torque equations for horizontal plane(no gravity) 
        t1 = fy*self.l1*np.cos(q1)-fx*self.l1*np.sin(q1)
        t2 = fy*self.l2*np.cos(q2)-fx*self.l2*np.sin(q2)

        b=0.3 #Damping coefficient

        dydt=[q1dot , q2dot , -t1/self.I1-b*q1dot , -t2/self.I2-b*q2dot]
        
        
        return dydt

    def step(self, dt):
        state = [self.q1, self.q2, self.q1dot , self.q2dot]

        self.ode.set_initial_value(state, 0)
        newstate = self.ode.integrate(dt)

        self.q1=newstate[0]
        self.q2=newstate[1]

        self.q1dot=newstate[2]
        self.q2dot=newstate[3]

        
    def getInfo(self):
        info = {
            "Position" : self.pos
        }
        return info

    def draw(self, image):
       

        j1 = (int(300+self.l1 * np.cos(-self.q1-np.pi/4)) , int(500+self.l1 * np.sin(-self.q1-np.pi/4)))

        j2 = (int(j1[0] + self.l2 * np.cos(-self.q2-np.pi/4)) , int(j1[1] + self.l2 * np.sin(-self.q2-np.pi/4)))

        self.pos=j2

        cv2.line(image,(300,500), j1, (0,255,0), 2)
        cv2.line(image,j1, j2, (0,0,255), 2)
        cv2.circle(image,j2,10,(255,0,0),-1)

        return image


robot = Robot(100,200,100,100,recordLocation = 'Task3_animation2.mp4')


for i in range(500):
    robot.step(0.1)
    robot.render()
    