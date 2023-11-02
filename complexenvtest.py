import epamodule as en
import numpy as np
import gym

from gym import spaces
from gym.spaces import Discrete, Box



#Environment definition:
#-Initiation
#-Step
#-Reset

class WaterSupplyEnv:
    #Definition of initialization parameters
    def __init__(self):
       #Opening the simulator "epamodule"
        en.ENopen("Bomba-deposito_v1_test.inp", "report.rpt", " ")




        #Definition of boundaries for the number of duty cycles
        self.counter=0
        self.duty_cycle = 5

        #Loading of tariff dataset
        self.numero = 1
        self.mes=6
        file_structure = open(file="marginalpdbcpt_20220"+str(self.mes)+'0' + str(self.numero) + ".1")
        file_structure = file_structure.read()
        lines = file_structure.split("\n")
        # Initialize an empty list to store the last floats
        self.tarifario1=[]

        # Iterate over each line and extract the last float
        for line in lines[1:25]:  # Exclude the first and last lines

            last_float = float(line.split(";")[4])*0.001
            self.tarifario1.append(last_float)

        #normalization of the tariff dataset
        divisor = max(self.tarifario1)
        self.tarifario = [ x/ divisor for x in self.tarifario1]

        #Definition of state features and initial paramenters
        en.ENopenH()
        self.tank,self.res,self.tas = self.node_index()
        self.pump,self.pipe= self.link_index()
        self.tanklevel = en.ENgetnodevalue(self.tank[0], en.EN_TANKLEVEL)
        self.pump_state = en.ENgetlinkvalue(self.pump[0], en.EN_STATUS)
        self.state1 = [self.tanklevel,self.pump_state,self.tarifario[0]*10,self.tarifario[0]*10,self.tarifario[1]*10,self.tarifario[2]*10,self.tarifario[3]*10,self.tarifario[4]*10,self.tarifario[5]*10,self.tarifario[6]*10,self.tarifario[7]*10,self.tarifario[8]*10,self.tarifario[9]*10,self.tarifario[10]*10,self.tarifario[11]*10,self.tarifario[12]*10,self.tarifario[13]*10,self.tarifario[14]*10,self.tarifario[15]*10,self.tarifario[16]*10,self.tarifario[17]*10,self.tarifario[18]*10,self.tarifario[19]*10,self.tarifario[20]*10,self.tarifario[21]*10,self.tarifario[22]*10,self.tarifario[23]*10,self.stepsize,self.counter,0,0,0]
        self.action_space =Discrete(2)  # pump position

        #Definition of box size domain for the state features
        limitesup=[10.0,1.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,50,50,50,100,100]
        limiteinf=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0,0,-50,0,0]

       # Loading Demand patterns

        self.demandqvc_file = "QVCdemands.txt"
        self.demandr_file = "rdemands.txt"

        file_structure = open(file=self.demandr_file)
        file_structure = file_structure.read()
        lines = file_structure.split("\n")

        # Load demand in node R
        self.demandr = []

        for i in range(48):
            a = lines[0].split(",")
            self.demandr.append(float(a[i]))
            self.state1.append(float(a[i]))
            #Definition of limits for the demands
            limitesup.append(100)
            limiteinf.append(0)
        en.ENsetpattern(2, self.demandr)

        #normalize demand R
        self.demandrnorm = []
        for i in range(48):
            self.demandrnorm.append(self.demandr[i] / max(self.demandr))
        self.demandr = self.demandrnorm

       #Load demand in node Vc
        file_structure2 = open(file=self.demandqvc_file)
        file_structure2 = file_structure2.read()
        lines2 = file_structure2.split("\n")

        self.demandqvc = []
        for i in range(48):
            x = lines2[0].split(",")
            self.demandqvc.append(float(x[i]))
            self.state1.append(float(x[i]))
            #Definition of limits for the demand
            limitesup.append(100)
            limiteinf.append(0)
        en.ENsetpattern(1, self.demandqvc)
        #normalization of demand in node Vc
        self.demandqvcnorm = []
        for i in range(48):
            self.demandqvcnorm.append(self.demandqvc[i] / max(self.demandqvc))
        self.demandqvc = self.demandqvcnorm
        self.state = np.array(self.state1)


        self.stepsize = 0
        self.parts=2
        self.states=[self.state[[0,1,2]]]
        self.cost=0
        self.observation_space=Box(low=np.array(limiteinf),high=np.array(limitesup),dtype=np.float32)

        en.ENinitH()
    #Definition of how steps procede in the environment and how the agent interacts with it
    def step(self, action):

        # Apply the action to the network
        pre_pump_state = en.ENgetlinkvalue(self.pump[0], en.EN_STATUS)
        pre_tank_level= en.ENgetnodevalue(self.tank[0], en.EN_PRESSURE)
        en.ENsetlinkvalue(self.pump[0],en.EN_STATUS,action)



        #Run action in simulation
        en.ENrunH()



        # Get the new state
        self.newtanklevel=en.ENgetnodevalue(self.tank[0],en.EN_PRESSURE)
        new_pump_state=en.ENgetlinkvalue(self.pump[0],en.EN_STATUS)
        energy=en.ENgetlinkvalue(self.pump[0],en.EN_ENERGY)
        self.energy=energy
        flow=self.newtanklevel-pre_tank_level

       #Calculation of Rewards
        reward=0
        #Reward for the tank level boundaries
        if 2>self.newtanklevel or self.newtanklevel>8:
            reward+=-100
        #Counter update
        if new_pump_state!=pre_pump_state and pre_pump_state==1:
            self.counter+=1
        #Reward for the limit of duty cycles
        if self.counter>self.duty_cycle:
            reward+=-100

        #Reward for the tariff price, calculation of energy cost and concatenation of state features
        if self.stepsize<1*self.parts:
            self.cost += (self.tarifario1[0]) * energy/self.parts
            new_state = [self.newtanklevel, new_pump_state,self.tarifario[0]*10,self.tarifario[0]*10,self.tarifario[1]*10,self.tarifario[2]*10,self.tarifario[3]*10,self.tarifario[4]*10,self.tarifario[5]*10,self.tarifario[6]*10,self.tarifario[7]*10,self.tarifario[8]*10,self.tarifario[9]*10,self.tarifario[10]*10,self.tarifario[11]*10,self.tarifario[12]*10,self.tarifario[13]*10,self.tarifario[14]*10,self.tarifario[15]*10,self.tarifario[16]*10,self.tarifario[17]*10,self.tarifario[18]*10,self.tarifario[19]*10,self.tarifario[20]*10,self.tarifario[21]*10,self.tarifario[22]*10,self.tarifario[23]*10,self.stepsize,self.counter,flow,self.demandr[step],self.demandqvc[step]]
            if new_pump_state==1:
                 reward+=-10**self.tarifario[0]
        if self.stepsize>=1*self.parts and self.stepsize<2*self.parts :
            self.cost += (self.tarifario1[1]) * energy/self.parts
            if new_pump_state==1:
                 reward+=-10**self.tarifario[1]
            new_state = [self.newtanklevel, new_pump_state, self.tarifario[1]*10,self.tarifario[0]*10,self.tarifario[1]*10,self.tarifario[2]*10,self.tarifario[3]*10,self.tarifario[4]*10,self.tarifario[5]*10,self.tarifario[6]*10,self.tarifario[7]*10,self.tarifario[8]*10,self.tarifario[9]*10,self.tarifario[10]*10,self.tarifario[11]*10,self.tarifario[12]*10,self.tarifario[13]*10,self.tarifario[14]*10,self.tarifario[15]*10,self.tarifario[16]*10,self.tarifario[17]*10,self.tarifario[18]*10,self.tarifario[19]*10,self.tarifario[20]*10,self.tarifario[21]*10,self.tarifario[22]*10,self.tarifario[23]*10,self.stepsize,self.counter,flow,self.demandr[step],self.demandqvc[step]]
        if self.stepsize>=2*self.parts and self.stepsize<3*self.parts :
            self.cost += (self.tarifario1[2]) * energy/self.parts
            if new_pump_state==1:
                 reward+=-10**self.tarifario[2]
            new_state = [self.newtanklevel, new_pump_state, self.tarifario[2]*10,self.tarifario[0]*10,self.tarifario[1]*10,self.tarifario[2]*10,self.tarifario[3]*10,self.tarifario[4]*10,self.tarifario[5]*10,self.tarifario[6]*10,self.tarifario[7]*10,self.tarifario[8]*10,self.tarifario[9]*10,self.tarifario[10]*10,self.tarifario[11]*10,self.tarifario[12]*10,self.tarifario[13]*10,self.tarifario[14]*10,self.tarifario[15]*10,self.tarifario[16]*10,self.tarifario[17]*10,self.tarifario[18]*10,self.tarifario[19]*10,self.tarifario[20]*10,self.tarifario[21]*10,self.tarifario[22]*10,self.tarifario[23]*10,self.stepsize,self.counter,flow,self.demandr[step],self.demandqvc[step]]
        if self.stepsize >=3*self.parts and self.stepsize<4*self.parts :
            self.cost += (self.tarifario1[3]) * energy/self.parts
            if new_pump_state==1:
                 reward+=-10**self.tarifario[3]
            new_state = [self.newtanklevel, new_pump_state, self.tarifario[3]*10,self.tarifario[0]*10,self.tarifario[1]*10,self.tarifario[2]*10,self.tarifario[3]*10,self.tarifario[4]*10,self.tarifario[5]*10,self.tarifario[6]*10,self.tarifario[7]*10,self.tarifario[8]*10,self.tarifario[9]*10,self.tarifario[10]*10,self.tarifario[11]*10,self.tarifario[12]*10,self.tarifario[13]*10,self.tarifario[14]*10,self.tarifario[15]*10,self.tarifario[16]*10,self.tarifario[17]*10,self.tarifario[18]*10,self.tarifario[19]*10,self.tarifario[20]*10,self.tarifario[21]*10,self.tarifario[22]*10,self.tarifario[23]*10,self.stepsize,self.counter,flow,self.demandr[step],self.demandqvc[step]]
        if self.stepsize >=4*self.parts and self.stepsize<5*self.parts:
            self.cost += (self.tarifario1[4]) * energy/self.parts
            if new_pump_state==1:
                 reward+=-10**self.tarifario[4]
            new_state = [self.newtanklevel, new_pump_state, self.tarifario[4]*10,self.tarifario[0]*10,self.tarifario[1]*10,self.tarifario[2]*10,self.tarifario[3]*10,self.tarifario[4]*10,self.tarifario[5]*10,self.tarifario[6]*10,self.tarifario[7]*10,self.tarifario[8]*10,self.tarifario[9]*10,self.tarifario[10]*10,self.tarifario[11]*10,self.tarifario[12]*10,self.tarifario[13]*10,self.tarifario[14]*10,self.tarifario[15]*10,self.tarifario[16]*10,self.tarifario[17]*10,self.tarifario[18]*10,self.tarifario[19]*10,self.tarifario[20]*10,self.tarifario[21]*10,self.tarifario[22]*10,self.tarifario[23]*10,self.stepsize,self.counter,flow,self.demandr[step],self.demandqvc[step]]
        if self.stepsize >=5*self.parts and self.stepsize<6*self.parts:
            self.cost += (self.tarifario1[5]) * energy/self.parts
            if new_pump_state==1:
                 reward+=-10**self.tarifario[5]
            new_state = [self.newtanklevel, new_pump_state, self.tarifario[5]*10,self.tarifario[0]*10,self.tarifario[1]*10,self.tarifario[2]*10,self.tarifario[3]*10,self.tarifario[4]*10,self.tarifario[5]*10,self.tarifario[6]*10,self.tarifario[7]*10,self.tarifario[8]*10,self.tarifario[9]*10,self.tarifario[10]*10,self.tarifario[11]*10,self.tarifario[12]*10,self.tarifario[13]*10,self.tarifario[14]*10,self.tarifario[15]*10,self.tarifario[16]*10,self.tarifario[17]*10,self.tarifario[18]*10,self.tarifario[19]*10,self.tarifario[20]*10,self.tarifario[21]*10,self.tarifario[22]*10,self.tarifario[23]*10,self.stepsize,self.counter,flow,self.demandr[step],self.demandqvc[step]]
        if self.stepsize >=6*self.parts and self.stepsize<7*self.parts:
            self.cost += (self.tarifario1[6]) * energy/self.parts
            if new_pump_state==1:
                 reward+=-10**self.tarifario[6]
            new_state = [self.newtanklevel, new_pump_state, self.tarifario[6]*10,self.tarifario[0]*10,self.tarifario[1]*10,self.tarifario[2]*10,self.tarifario[3]*10,self.tarifario[4]*10,self.tarifario[5]*10,self.tarifario[6]*10,self.tarifario[7]*10,self.tarifario[8]*10,self.tarifario[9]*10,self.tarifario[10]*10,self.tarifario[11]*10,self.tarifario[12]*10,self.tarifario[13]*10,self.tarifario[14]*10,self.tarifario[15]*10,self.tarifario[16]*10,self.tarifario[17]*10,self.tarifario[18]*10,self.tarifario[19]*10,self.tarifario[20]*10,self.tarifario[21]*10,self.tarifario[22]*10,self.tarifario[23]*10,self.stepsize,self.counter,flow,self.demandr[step],self.demandqvc[step]]
        if self.stepsize >=7*self.parts and self.stepsize<8*self.parts:
            self.cost += (self.tarifario1[7]) * energy/self.parts
            if new_pump_state==1:
                 reward+=-10**self.tarifario[7]
            new_state = [self.newtanklevel, new_pump_state, self.tarifario[7]*10,self.tarifario[0]*10,self.tarifario[1]*10,self.tarifario[2]*10,self.tarifario[3]*10,self.tarifario[4]*10,self.tarifario[5]*10,self.tarifario[6]*10,self.tarifario[7]*10,self.tarifario[8]*10,self.tarifario[9]*10,self.tarifario[10]*10,self.tarifario[11]*10,self.tarifario[12]*10,self.tarifario[13]*10,self.tarifario[14]*10,self.tarifario[15]*10,self.tarifario[16]*10,self.tarifario[17]*10,self.tarifario[18]*10,self.tarifario[19]*10,self.tarifario[20]*10,self.tarifario[21]*10,self.tarifario[22]*10,self.tarifario[23]*10,self.stepsize,self.counter,flow,self.demandr[step],self.demandqvc[step]]
        if self.stepsize >=8*self.parts and self.stepsize<9*self.parts:
            self.cost += (self.tarifario1[8]) * energy/self.parts
            if new_pump_state==1:
                 reward+=-10**self.tarifario[8]
            new_state = [self.newtanklevel, new_pump_state, self.tarifario[8]*10,self.tarifario[0]*10,self.tarifario[1]*10,self.tarifario[2]*10,self.tarifario[3]*10,self.tarifario[4]*10,self.tarifario[5]*10,self.tarifario[6]*10,self.tarifario[7]*10,self.tarifario[8]*10,self.tarifario[9]*10,self.tarifario[10]*10,self.tarifario[11]*10,self.tarifario[12]*10,self.tarifario[13]*10,self.tarifario[14]*10,self.tarifario[15]*10,self.tarifario[16]*10,self.tarifario[17]*10,self.tarifario[18]*10,self.tarifario[19]*10,self.tarifario[20]*10,self.tarifario[21]*10,self.tarifario[22]*10,self.tarifario[23]*10,self.stepsize,self.counter,flow,self.demandr[step],self.demandqvc[step]]
        if self.stepsize >=9*self.parts and self.stepsize<10*self.parts:
            self.cost += (self.tarifario1[9]) * energy/self.parts
            if new_pump_state==1:
                 reward+=-10**self.tarifario[9]
            new_state = [self.newtanklevel, new_pump_state, self.tarifario[9]*10,self.tarifario[0]*10,self.tarifario[1]*10,self.tarifario[2]*10,self.tarifario[3]*10,self.tarifario[4]*10,self.tarifario[5]*10,self.tarifario[6]*10,self.tarifario[7]*10,self.tarifario[8]*10,self.tarifario[9]*10,self.tarifario[10]*10,self.tarifario[11]*10,self.tarifario[12]*10,self.tarifario[13]*10,self.tarifario[14]*10,self.tarifario[15]*10,self.tarifario[16]*10,self.tarifario[17]*10,self.tarifario[18]*10,self.tarifario[19]*10,self.tarifario[20]*10,self.tarifario[21]*10,self.tarifario[22]*10,self.tarifario[23]*10,self.stepsize,self.counter,flow,self.demandr[step],self.demandqvc[step]]
        if self.stepsize >=10*self.parts and self.stepsize<11*self.parts:
            self.cost += (self.tarifario1[10]) * energy/self.parts
            if new_pump_state==1:
                 reward+=-10**self.tarifario[10]
            new_state = [self.newtanklevel, new_pump_state, self.tarifario[10]*10,self.tarifario[0]*10,self.tarifario[1]*10,self.tarifario[2]*10,self.tarifario[3]*10,self.tarifario[4]*10,self.tarifario[5]*10,self.tarifario[6]*10,self.tarifario[7]*10,self.tarifario[8]*10,self.tarifario[9]*10,self.tarifario[10]*10,self.tarifario[11]*10,self.tarifario[12]*10,self.tarifario[13]*10,self.tarifario[14]*10,self.tarifario[15]*10,self.tarifario[16]*10,self.tarifario[17]*10,self.tarifario[18]*10,self.tarifario[19]*10,self.tarifario[20]*10,self.tarifario[21]*10,self.tarifario[22]*10,self.tarifario[23]*10,self.stepsize,self.counter,flow,self.demandr[step],self.demandqvc[step]]
        if self.stepsize >=11*self.parts and self.stepsize<12*self.parts:
            self.cost += (self.tarifario1[11]) * energy/self.parts
            if new_pump_state==1:
                 reward+=-10**self.tarifario[11]
            new_state = [self.newtanklevel, new_pump_state, self.tarifario[11]*10,self.tarifario[0]*10,self.tarifario[1]*10,self.tarifario[2]*10,self.tarifario[3]*10,self.tarifario[4]*10,self.tarifario[5]*10,self.tarifario[6]*10,self.tarifario[7]*10,self.tarifario[8]*10,self.tarifario[9]*10,self.tarifario[10]*10,self.tarifario[11]*10,self.tarifario[12]*10,self.tarifario[13]*10,self.tarifario[14]*10,self.tarifario[15]*10,self.tarifario[16]*10,self.tarifario[17]*10,self.tarifario[18]*10,self.tarifario[19]*10,self.tarifario[20]*10,self.tarifario[21]*10,self.tarifario[22]*10,self.tarifario[23]*10,self.stepsize,self.counter,flow,self.demandr[step],self.demandqvc[step]]
        if self.stepsize >=12*self.parts and self.stepsize<13*self.parts:
            self.cost += (self.tarifario1[12]) * energy/self.parts
            if new_pump_state==1:
                 reward+=-10**self.tarifario[12]
            new_state = [self.newtanklevel, new_pump_state, self.tarifario[12]*10,self.tarifario[0]*10,self.tarifario[1]*10,self.tarifario[2]*10,self.tarifario[3]*10,self.tarifario[4]*10,self.tarifario[5]*10,self.tarifario[6]*10,self.tarifario[7]*10,self.tarifario[8]*10,self.tarifario[9]*10,self.tarifario[10]*10,self.tarifario[11]*10,self.tarifario[12]*10,self.tarifario[13]*10,self.tarifario[14]*10,self.tarifario[15]*10,self.tarifario[16]*10,self.tarifario[17]*10,self.tarifario[18]*10,self.tarifario[19]*10,self.tarifario[20]*10,self.tarifario[21]*10,self.tarifario[22]*10,self.tarifario[23]*10,self.stepsize,self.counter,flow,self.demandr[step],self.demandqvc[step]]
        if self.stepsize >=13*self.parts and self.stepsize<14*self.parts:
            self.cost += (self.tarifario1[13]) * energy/self.parts
            if new_pump_state==1:
                 reward+=-10**self.tarifario[13]
            new_state = [self.newtanklevel, new_pump_state, self.tarifario[13]*10,self.tarifario[0]*10,self.tarifario[1]*10,self.tarifario[2]*10,self.tarifario[3]*10,self.tarifario[4]*10,self.tarifario[5]*10,self.tarifario[6]*10,self.tarifario[7]*10,self.tarifario[8]*10,self.tarifario[9]*10,self.tarifario[10]*10,self.tarifario[11]*10,self.tarifario[12]*10,self.tarifario[13]*10,self.tarifario[14]*10,self.tarifario[15]*10,self.tarifario[16]*10,self.tarifario[17]*10,self.tarifario[18]*10,self.tarifario[19]*10,self.tarifario[20]*10,self.tarifario[21]*10,self.tarifario[22]*10,self.tarifario[23]*10,self.stepsize,self.counter,flow,self.demandr[step],self.demandqvc[step]]
        if self.stepsize >=14*self.parts and self.stepsize<15*self.parts:
            self.cost += (self.tarifario1[14]) * energy/self.parts
            if new_pump_state==1:
                 reward+=-10**self.tarifario[14]
            new_state = [self.newtanklevel, new_pump_state, self.tarifario[14]*10,self.tarifario[0]*10,self.tarifario[1]*10,self.tarifario[2]*10,self.tarifario[3]*10,self.tarifario[4]*10,self.tarifario[5]*10,self.tarifario[6]*10,self.tarifario[7]*10,self.tarifario[8]*10,self.tarifario[9]*10,self.tarifario[10]*10,self.tarifario[11]*10,self.tarifario[12]*10,self.tarifario[13]*10,self.tarifario[14]*10,self.tarifario[15]*10,self.tarifario[16]*10,self.tarifario[17]*10,self.tarifario[18]*10,self.tarifario[19]*10,self.tarifario[20]*10,self.tarifario[21]*10,self.tarifario[22]*10,self.tarifario[23]*10,self.stepsize,self.counter,flow,self.demandr[step],self.demandqvc[step]]
        if self.stepsize >= 15 * self.parts and self.stepsize < 16 * self.parts:
            self.cost += (self.tarifario1[15]) * energy/self.parts
            if new_pump_state==1:
                 reward+=-10**self.tarifario[15]
            new_state = [self.newtanklevel, new_pump_state, self.tarifario[15] * 10,self.tarifario[0]*10,self.tarifario[1]*10,self.tarifario[2]*10,self.tarifario[3]*10,self.tarifario[4]*10,self.tarifario[5]*10,self.tarifario[6]*10,self.tarifario[7]*10,self.tarifario[8]*10,self.tarifario[9]*10,self.tarifario[10]*10,self.tarifario[11]*10,self.tarifario[12]*10,self.tarifario[13]*10,self.tarifario[14]*10,self.tarifario[15]*10,self.tarifario[16]*10,self.tarifario[17]*10,self.tarifario[18]*10,self.tarifario[19]*10,self.tarifario[20]*10,self.tarifario[21]*10,self.tarifario[22]*10,self.tarifario[23]*10,self.stepsize,self.counter,flow,self.demandr[step],self.demandqvc[step]]
        if self.stepsize >=16*self.parts and self.stepsize<17*self.parts:
            self.cost += (self.tarifario1[16]) * energy/self.parts
            if new_pump_state==1:
                 reward+=-10**self.tarifario[16]
            new_state = [self.newtanklevel, new_pump_state, self.tarifario[16]*10,self.tarifario[0]*10,self.tarifario[1]*10,self.tarifario[2]*10,self.tarifario[3]*10,self.tarifario[4]*10,self.tarifario[5]*10,self.tarifario[6]*10,self.tarifario[7]*10,self.tarifario[8]*10,self.tarifario[9]*10,self.tarifario[10]*10,self.tarifario[11]*10,self.tarifario[12]*10,self.tarifario[13]*10,self.tarifario[14]*10,self.tarifario[15]*10,self.tarifario[16]*10,self.tarifario[17]*10,self.tarifario[18]*10,self.tarifario[19]*10,self.tarifario[20]*10,self.tarifario[21]*10,self.tarifario[22]*10,self.tarifario[23]*10,self.stepsize,self.counter,flow,self.demandr[step],self.demandqvc[step]]
        if self.stepsize >=17*self.parts and self.stepsize<18*self.parts:
            self.cost += (self.tarifario1[17]) * energy/self.parts
            if new_pump_state==1:
                 reward+=-10**self.tarifario[17]
            new_state = [self.newtanklevel, new_pump_state, self.tarifario[17]*10,self.tarifario[0]*10,self.tarifario[1]*10,self.tarifario[2]*10,self.tarifario[3]*10,self.tarifario[4]*10,self.tarifario[5]*10,self.tarifario[6]*10,self.tarifario[7]*10,self.tarifario[8]*10,self.tarifario[9]*10,self.tarifario[10]*10,self.tarifario[11]*10,self.tarifario[12]*10,self.tarifario[13]*10,self.tarifario[14]*10,self.tarifario[15]*10,self.tarifario[16]*10,self.tarifario[17]*10,self.tarifario[18]*10,self.tarifario[19]*10,self.tarifario[20]*10,self.tarifario[21]*10,self.tarifario[22]*10,self.tarifario[23]*10,self.stepsize,self.counter,flow,self.demandr[step],self.demandqvc[step]]
        if self.stepsize >=18*self.parts and self.stepsize<19*self.parts:
            self.cost += (self.tarifario1[18]) * energy/self.parts
            if new_pump_state==1:
                 reward+=-10**self.tarifario[18]
            new_state = [self.newtanklevel, new_pump_state, self.tarifario[18]*10,self.tarifario[0]*10,self.tarifario[1]*10,self.tarifario[2]*10,self.tarifario[3]*10,self.tarifario[4]*10,self.tarifario[5]*10,self.tarifario[6]*10,self.tarifario[7]*10,self.tarifario[8]*10,self.tarifario[9]*10,self.tarifario[10]*10,self.tarifario[11]*10,self.tarifario[12]*10,self.tarifario[13]*10,self.tarifario[14]*10,self.tarifario[15]*10,self.tarifario[16]*10,self.tarifario[17]*10,self.tarifario[18]*10,self.tarifario[19]*10,self.tarifario[20]*10,self.tarifario[21]*10,self.tarifario[22]*10,self.tarifario[23]*10,self.stepsize,self.counter,flow,self.demandr[step],self.demandqvc[step]]
        if self.stepsize >=19*self.parts and self.stepsize<20*self.parts:
            self.cost += (self.tarifario1[19]) * energy/self.parts
            if new_pump_state==1:
                 reward+=-10**self.tarifario[19]
            new_state = [self.newtanklevel, new_pump_state, self.tarifario[19]*10,self.tarifario[0]*10,self.tarifario[1]*10,self.tarifario[2]*10,self.tarifario[3]*10,self.tarifario[4]*10,self.tarifario[5]*10,self.tarifario[6]*10,self.tarifario[7]*10,self.tarifario[8]*10,self.tarifario[9]*10,self.tarifario[10]*10,self.tarifario[11]*10,self.tarifario[12]*10,self.tarifario[13]*10,self.tarifario[14]*10,self.tarifario[15]*10,self.tarifario[16]*10,self.tarifario[17]*10,self.tarifario[18]*10,self.tarifario[19]*10,self.tarifario[20]*10,self.tarifario[21]*10,self.tarifario[22]*10,self.tarifario[23]*10,self.stepsize,self.counter,flow,self.demandr[step],self.demandqvc[step]]
        if self.stepsize >=20*self.parts and self.stepsize<21*self.parts:
            self.cost += (self.tarifario1[20]) * energy/self.parts
            if new_pump_state==1:
                 reward+=-10**self.tarifario[20]
            new_state = [self.newtanklevel, new_pump_state, self.tarifario[20]*10,self.tarifario[0]*10,self.tarifario[1]*10,self.tarifario[2]*10,self.tarifario[3]*10,self.tarifario[4]*10,self.tarifario[5]*10,self.tarifario[6]*10,self.tarifario[7]*10,self.tarifario[8]*10,self.tarifario[9]*10,self.tarifario[10]*10,self.tarifario[11]*10,self.tarifario[12]*10,self.tarifario[13]*10,self.tarifario[14]*10,self.tarifario[15]*10,self.tarifario[16]*10,self.tarifario[17]*10,self.tarifario[18]*10,self.tarifario[19]*10,self.tarifario[20]*10,self.tarifario[21]*10,self.tarifario[22]*10,self.tarifario[23]*10,self.stepsize,self.counter,flow,self.demandr[step],self.demandqvc[step]]
        if self.stepsize >=21*self.parts and self.stepsize<22*self.parts:
            self.cost += (self.tarifario1[21]) * energy/self.parts
            if new_pump_state==1:
                 reward+=-10**self.tarifario[21]
            new_state = [self.newtanklevel, new_pump_state, self.tarifario[21]*10,self.tarifario[0]*10,self.tarifario[1]*10,self.tarifario[2]*10,self.tarifario[3]*10,self.tarifario[4]*10,self.tarifario[5]*10,self.tarifario[6]*10,self.tarifario[7]*10,self.tarifario[8]*10,self.tarifario[9]*10,self.tarifario[10]*10,self.tarifario[11]*10,self.tarifario[12]*10,self.tarifario[13]*10,self.tarifario[14]*10,self.tarifario[15]*10,self.tarifario[16]*10,self.tarifario[17]*10,self.tarifario[18]*10,self.tarifario[19]*10,self.tarifario[20]*10,self.tarifario[21]*10,self.tarifario[22]*10,self.tarifario[23]*10,self.stepsize,self.counter,flow,self.demandr[step],self.demandqvc[step]]
        if self.stepsize >=22*self.parts and self.stepsize<23*self.parts:
            self.cost += (self.tarifario1[22]) * energy/self.parts
            if new_pump_state==1:
                 reward+=-10**self.tarifario[22]
            new_state = [self.newtanklevel, new_pump_state, self.tarifario[22] * 10,self.tarifario[0]*10,self.tarifario[1]*10,self.tarifario[2]*10,self.tarifario[3]*10,self.tarifario[4]*10,self.tarifario[5]*10,self.tarifario[6]*10,self.tarifario[7]*10,self.tarifario[8]*10,self.tarifario[9]*10,self.tarifario[10]*10,self.tarifario[11]*10,self.tarifario[12]*10,self.tarifario[13]*10,self.tarifario[14]*10,self.tarifario[15]*10,self.tarifario[16]*10,self.tarifario[17]*10,self.tarifario[18]*10,self.tarifario[19]*10,self.tarifario[20]*10,self.tarifario[21]*10,self.tarifario[22]*10,self.tarifario[23]*10,self.stepsize,self.counter,flow,self.demandr[step],self.demandqvc[step]]
        if self.stepsize>=23*self.parts :
            self.cost += (self.tarifario1[23]) * energy/self.parts
            if new_pump_state==1:
                 reward+=-10**self.tarifario[23]
            new_state = [self.newtanklevel, new_pump_state, self.tarifario[23]*10,self.tarifario[0]*10,self.tarifario[1]*10,self.tarifario[2]*10,self.tarifario[3]*10,self.tarifario[4]*10,self.tarifario[5]*10,self.tarifario[6]*10,self.tarifario[7]*10,self.tarifario[8]*10,self.tarifario[9]*10,self.tarifario[10]*10,self.tarifario[11]*10,self.tarifario[12]*10,self.tarifario[13]*10,self.tarifario[14]*10,self.tarifario[15]*10,self.tarifario[16]*10,self.tarifario[17]*10,self.tarifario[18]*10,self.tarifario[19]*10,self.tarifario[20]*10,self.tarifario[21]*10,self.tarifario[22]*10,self.tarifario[23]*10,self.stepsize,self.counter,flow,self.demandr[step],self.demandqvc[step]]

        for i in range(48):
            new_state.append(self.demandqvc[i])
        for i in range(48):
            new_state.append(self.demandr[i])

        new_state = np.array(new_state)

        self.states.append(new_state[[0,1,2]])

        # CHECK IF DONE
        ts = en.ENnextH()
        done = self.is_done(ts)

        #Update time-step
        self.stepsize += 1
        info={}

        return new_state, reward, done, info

    # Reset the network and get the initial state. The contents are similar to initialization
    #although there are changes in tariff and demand patterns
    def reset(self):
        #Restart simulator
        en.ENcloseH()
        en.ENclose()
        en.ENopen("Bomba-deposito_v1_test.inp", "report.rpt", " ")


        #Change of Tariff pattern
        self.numero += 1
        if self.numero>28 and self.mes==2:
            self.mes+=1
            self.numero=1
        if self.numero>30:
            self.mes+=1
            self.numero=1
            if self.mes > 12:
                self.mes=1
        if self.numero<10 and self.mes<10:
            file_structure = open(file="marginalpdbcpt_20220"+str(self.mes)+'0' + str(self.numero) + ".1")
            file_structure = file_structure.read()
            lines = file_structure.split("\n")
            # Initialize an empty list to store the last floats
        if self.numero>=10 and self.mes>=10:
            file_structure = open(file="marginalpdbcpt_2022"+str(self.mes) + str(self.numero) + ".1")
            file_structure = file_structure.read()
            lines = file_structure.split("\n")
        if self.numero >= 10 and self.mes <10:
            file_structure = open(file="marginalpdbcpt_20220" + str(self.mes) + str(self.numero) + ".1")
            file_structure = file_structure.read()
            lines = file_structure.split("\n")

            # Initialize an empty list to store the last floats
        if self.numero<10 and self.mes>=10:
            file_structure = open(file="marginalpdbcpt_2022"+str(self.mes)+'0' + str(self.numero) + ".1")
            file_structure = file_structure.read()
            lines = file_structure.split("\n")

            # Iterate over each line and extract the last float
        self.tarifario1= []
        for line in lines[1:25]:  # Exclude the first and last lines

            last_float = float(line.split(";")[4])*0.001
            self.tarifario1.append(last_float)

        #normalize tariff pattern
        divisor = max(self.tarifario1)


        self.tarifario = [x / divisor for x in self.tarifario1]




        en.ENopenH()
        self.tank, self.res, self.tas = self.node_index()
        self.pump, self.pipe = self.link_index()
        self.counter = 0
        self.tanklevel = en.ENgetnodevalue(self.tank[0], en.EN_TANKLEVEL)
        self.pump_state = 0 #en.ENgetlinkvalue(self.pump[0], en.EN_STATUS)
        self.state1 = ([self.tanklevel, self.pump_state, self.tarifario[0]*10,self.tarifario[0]*10,self.tarifario[1]*10,self.tarifario[2]*10,self.tarifario[3]*10,self.tarifario[4]*10,self.tarifario[5]*10,self.tarifario[6]*10,self.tarifario[7]*10,self.tarifario[8]*10,self.tarifario[9]*10,self.tarifario[10]*10,self.tarifario[11]*10,self.tarifario[12]*10,self.tarifario[13]*10,self.tarifario[14]*10,self.tarifario[15]*10,self.tarifario[16]*10,self.tarifario[17]*10,self.tarifario[18]*10,self.tarifario[19]*10,self.tarifario[20]*10,self.tarifario[21]*10,self.tarifario[22]*10,self.tarifario[23]*10,self.stepsize,self.counter,0,0,0])


        #update demand patterns
        self.day+=1
        if self.day == 360:
            self.day = 0
        file_structure = open(file=self.demandr_file)
        file_structure = file_structure.read()
        lines = file_structure.split("\n")
        self.demandr = []
        # Iterate over each line and extract the last float
        for i in range(48):  # Exclude the first and last lines
            a = lines[self.day].split(",")
            self.demandr.append(float(a[i]))
            self.state1.append(float(a[i]))
        en.ENsetpattern(2, self.demandr)
        self.demandrnorm = []
        for i in range(48):
            self.demandrnorm.append(self.demandr[i] / max(self.demandr))
        self.demandr = self.demandrnorm

        file_structure2 = open(file=self.demandqvc_file)
        file_structure2 = file_structure2.read()
        lines2 = file_structure2.split("\n")

        self.demandqvc = []
        for i in range(48):
            x = lines2[self.day].split(",")
            self.demandqvc.append(float(x[i]))
            self.state1.append(float(x[i]))
        en.ENsetpattern(1, self.demandqvc)

        self.demanqvcnorm = []
        for i in range(48):
            self.demanqvcnorm.append(self.demandqvc[i] / max(self.demandqvc))
        self.demandqvc = self.demandqvcnorm

        self.state = np.array(self.state1)
        self.stepsize = 0

        self.states = [self.state[[0,1,2]]]

        self.cost = 0

        en.ENinitH()
        return self.state

    def calculate_reward(self, new_state):
        # Calculate the reward based on the new state
        return 1.0  # Placeholder reward function

    def is_done(self,ts):
        if ts>0:
            return False  # Placeholder episode termination condition
        else:
            #Although the simulation is finished, it still counts an extra step. This conditions
            #eliminate this error.
            self.states = self.states[: -1]
            self.cost -= (self.tarifario1[23]) * self.energy / self.parts

            return True


 #Get node index
    def node_index(self):
        n_nodes = en.ENgetcount(en.EN_NODECOUNT)
        tank_idx = []
        junction_idx = []
        reservoir_idx = []
        for i in range(1, n_nodes + 1):
            type = en.ENgetnodetype(i)
            if type == en.EN_TANK:
                tank_idx.append(i)
            if type == en.EN_JUNCTION:
                junction_idx.append(i)
            if type == en.EN_RESERVOIR:
                reservoir_idx.append(i)

        return tank_idx, junction_idx, reservoir_idx

#Get links Indexs
    def link_index(self):
        n_links = en.ENgetcount(en.EN_LINKCOUNT)
        pump_idx = []
        pipe_idx = []
        for i in range(1, n_links + 1):
            type = en.ENgetlinktype(i)
            if type == en.EN_PUMP:
                pump_idx.append(i)
            if type == en.EN_PIPE:
                pipe_idx.append(i)


        return pump_idx, pipe_idx