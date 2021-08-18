import numpy as np
import matplotlib.pyplot as plt

class Robot():
    def __init__(self):
        super().__init__()
        self.q1=35*np.pi/180
        self.q2=35*np.pi/180
        self.l1=300
        self.l2=200
        self.X=[]
        self.Y=[]

        
    
    def step(self):
        x = self.l1*np.cos(self.q1) + self.l2*np.cos(self.q2)
        y = self.l1*np.sin(self.q1) + self.l2*np.sin(self.q2)

        self.X.append(x)
        self.Y.append(y)


robot = Robot()

while(robot.q1<=145*np.pi/180):
    while(robot.q2<=145*np.pi/180):
        robot.step()
        robot.q2+=0.02
    
    robot.q1+=0.02
    robot.q2=35*np.pi/180


plt.scatter(robot.X,robot.Y)
plt.title("Workspace- 2R (45 deg to 135 deg)")
plt.xlabel("X Co-ordinate")
plt.ylabel("Y Co-ordinate")
plt.show()

