# establish a GA optimizer
import inspyred
from random import Random, seed
import time 
import os
import pickle 
import numpy as np 
import gym
from gym import wrappers
from inspyred.ec import emo
import shishi_gym
import CDA42_gym
import CDA44_gym

from huatu import huatu
from transfer import transfer

# this is for surrogate.
# there would be 'cannot attribute Surrogate' without these lines.
import sys 
sys.path.append(r'C:/Users/y/Desktop/DDPGshishi/DDPG-master/CDA42-gym/KrigingPython')
from Surrogate_01de import Surrugate

class GAoptimizer():
    def __init__(self, ENV_NAME):
        self.name = 'GA optimizer'
        self.ENV_NAME=ENV_NAME
        self.environment = gym.make(ENV_NAME)
        self.state_dim = self.environment.observation_space.shape[0]
        self.action_dim = self.environment.action_space.shape[0]

        self.location = 'C:/Users/y/Desktop/GApython/results'
        shijian = time.strftime("%Y-%m-%d", time.localtime())
        self.save_location = self.location + '/GAresults' + shijian

        self.transfer = transfer(tishi=0)

        try:
            os.mkdir(self.save_location)
        except:
            print('MXairfoil: folder already there')

        self.final_pop = 0
        self.N_call_env = 0 

        rand = Random()
        # rand.seed(int(time.time()))
        rand.seed(time.time())
        self.ea = inspyred.ec.emo.NSGA2(rand)
        self.ea.terminator = inspyred.ec.terminators.generation_termination
        self.ea.variator = [inspyred.ec.variators.blend_crossover, inspyred.ec.variators.gaussian_mutation]

        self.converge_history = np.array([0,0]).reshape(1,2)
        self.converge_history_last = 0

        self.candidates = 0 
        self.fitness = 0 

        self.step_number = 100 
        self.pop_size = 100 
        self.max_generations = 400

    def generate_sin(self,random, args):
        size = args.get('num_inputs', 4) # if num_input are not setted 
        return [random.uniform(-1,1) for i in range(size)]

    def generate_env(self,random, args):
        size = args.get('num_inputs', 4) # if num_input are not setted 
        xiajie = self.environment.observation_space.low[0]
        shangjie = self.environment.observation_space.high[0]
        return [random.uniform(xiajie,shangjie) for i in range(size)]
    '''
    def evaluate_env(self,candidates, args):
        # this is one way to move.
        fitness = []
        env = gym.make(self.ENV_NAME)
        env.reset()
        for cs in candidates:
            cs_sign = np.sign(cs)
            cs_abs = np.abs(cs)
            
            while cs_abs >0.1 :
                next_state,reward,done,_ = env.step(0.1*cs_sign)
                cs_abs = cs_abs -0.1
            next_state,reward,done,_ = env.step(cs_abs*cs_sign)
            fitness.append(np.array(reward).reshape(1,))
            env.reset()
        del env
        return fitness

    def evaluate_env2(self,candidates, args):
        # this is another way to move, which is more yangjian.
        fitness = []
        env = gym.make(self.ENV_NAME)
        env.reset()
        for cs in candidates:
            cs_max = np.max(np.abs(cs)) 
            d_step = 0.1
            N = int(round(cs_max/d_step))
            cs_sign = np.sign(cs)
            for i in range(N):
                next_state,reward,done,asd = env.step(d_step*cs_sign)
            next_state,reward,done,asd = env.step((np.array(cs) - d_step*N*cs_sign))
            fitness.append(reward)
            env.reset()
        del env
        return fitness
    '''
    def evaluate_env3(self,candidates, args):
        # this is for CDA42, in which there are set_state
        fitness = []
        # env = gym.make(self.ENV_NAME)
        env = self.environment
        # self.N_call_env = self.N_call_env + env.N_step 
        env.reset()
        # env = self.environment # since it is jiade parallel, use one env maybe dangerous 
        for cs in candidates:
            next_state,reward,done,asd = env.set_state(cs)
            self.N_call_env = self.N_call_env +1 
            fitness.append(emo.Pareto([reward]))
            env.reset()
        self.record_converge_history(self.N_call_env,np.max(fitness))
        rizhi = 'MXairfoil: finish calculating one generation, N_call_env = '+str(self.N_call_env)+'\nmax fitness ='+str(np.max(fitness))
        self.jilu(rizhi)
        return fitness

    def evaluate_env4(self,candidates, args):
        # this is for CDA44, in which there are set_state, and 4dim with constrain.
        fitness = []
        # env = gym.make(self.ENV_NAME)
        env = self.environment
        # self.N_call_env = self.N_call_env + env.N_step 
        env.reset()
        # env = self.environment # since it is jiade parallel, use one env maybe dangerous 
        for cs in candidates:
            next_state,reward,done,asd = env.set_state(cs)
            self.N_call_env = self.N_call_env +1 
            fitness.append(emo.Pareto([reward]))
            env.reset()
        self.record_converge_history(self.N_call_env,np.max(fitness))
        rizhi = 'MXairfoil: finish calculating one generation, N_call_env = '+str(self.N_call_env)+'\nmax fitness ='+str(np.max(fitness))
        self.jilu(rizhi)
        return fitness

    def evaluate_sin(self,candidates, args):
        fitness = []
        for cs in candidates:
            fit = np.sin(np.pi*cs[0])
            # fitness.append(np.array(fit))
            fitness.append(emo.Pareto([fit]))
            self.N_call_env = self.N_call_env+1
            if self.N_call_env % 100 == 99 :
                print('MXairfoil: evaluating sin function, N_call_env='+str(self.N_call_env))
        self.jilu('shishi, N_call_env = '+str(self.N_call_env) )
        return fitness

    def generate_policy(self,random,args):
        step_number = self.step_number 
        action_dim = self.action_dim
        size = args.get('num_inputs', step_number*action_dim) 
        xiajie = self.environment.action_space.low[0]
        shangjie = self.environment.action_space.high[0]
        return [random.uniform(xiajie,shangjie) for i in range(size)]

    def evaluate_policy(self,candidates, args):
        # this is for certain policy, 2 dim, 100 steps, CDA42 
        step_number = self.step_number 
        dim = self.action_dim
        pop_size=self.pop_size
        max_generations=self.max_generations
        fitness = []
        env = self.environment
        
        for cs in candidates:
            state = env.reset_original() # start from original point.
            # cs includes all actions
            total_reward = 0 
            for j in range(step_number):
                action = cs[dim*j : dim*(j+1)]
                state,reward,done,_ = env.step(action)
                total_reward += reward
                self.N_call_env = self.N_call_env +1 
                if self.N_call_env%1000 == 999:
                    print('MXairfoil: calculating, N_call_env = ' +str(self.N_call_env)+'\nbili: '+str(self.N_call_env/(step_number*pop_size*max_generations)*100)+'%'+ '\nSaving the intermediat result')
                    

            average_reward = total_reward / step_number
            fitness.append(emo.Pareto([average_reward]))
        
        self.candidates = candidates
        self.fitness = fitness
        self.record_converge_history(self.N_call_env,np.max(fitness))
        rizhi = 'MXairfoil: finish calculating one generation, N_call_env = '+str(self.N_call_env)+'\nmax fitness ='+str(np.max(fitness))+'\n result saved'
        self.save_GA()
        self.jilu(rizhi)
        return fitness

    def train_test(self):
        final_pop = self.ea.evolve(generator=self.generate_sin,evaluator=self.evaluate_sin,pop_size=100,maximize=True,bounder=inspyred.ec.Bounder(-1, 1),max_evaluations=20000,mutation_rate=0.25,num_inputs=1)
        return final_pop

    def train_env(self):
        print('MXairfoil: start to train something interesting')

        projdir = os.path.dirname(os.getcwd())
        stat_file_name = self.location + '/stat_file.csv'
        ind_file_name = self.location + '/ind_file_name.csv'
        stat_file = open(stat_file_name, 'w')
        ind_file = open(ind_file_name, 'w')
        
        
        if self.ENV_NAME == 'CDA42_env-v0':
            final_pop = self.ea.evolve(generator=self.generate_env,evaluator=self.evaluate_env3,pop_size=100,seed=[],maximize=True,bounder=inspyred.ec.Bounder([-1,-1],[1,1]),max_generations=3,mutation_rate=0.25,num_inputs=2,statistics_file=stat_file,individuals_file=ind_file)
        elif self.ENV_NAME == 'CDA44_env-v0':
            final_pop = self.ea.evolve(generator=self.generate_env,evaluator=self.evaluate_env4,pop_size=100,seed=[],maximize=True,bounder=inspyred.ec.Bounder([-1,-1,-1,-1],[1,1,1,1]),max_generations=200,mutation_rate=0.25,num_inputs=4,statistics_file=stat_file,individuals_file=ind_file)

        stat_file.close()
        ind_file.close()

        final_pop.sort(reverse=True)
        self.final_pop = final_pop
        # self.converge_history = np.array([self.N_call_env,final_pop[0].fitness])
        rizhi = '\n\nMXairfoil: (GA) finished, final_pop[0] = '+str(final_pop[0])+'\n converge in '+str(self.converge_history)
        self.jilu(rizhi)
        return final_pop

    def train_policy(self):
        print('MXairfoil: start to train policy from original points')
        projdir = os.path.dirname(os.getcwd())
        step_number = self.step_number 
        action_dim = self.action_dim

        # shangjie = np.ones(step_number*action_dim)
        # xiajie = shangjie*-1
        shangjie0 = [1.]
        xiajie0 = [-1.]
        shangjie = []
        xiajie = [] 
        for i in range(step_number*action_dim):
            shangjie = shangjie + shangjie0
            xiajie = xiajie + xiajie0


        if self.ENV_NAME == 'CDA42_env-v0':
            final_pop = self.ea.evolve(generator=self.generate_policy,evaluator=self.evaluate_policy,pop_size=self.pop_size,seed=[],maximize=True,bounder=inspyred.ec.Bounder(xiajie,shangjie),max_generations=self.max_generations,mutation_rate=0.05,num_inputs=step_number*action_dim)
        elif self.ENV_NAME == 'CDA44_env-v0':
            final_pop = self.ea.evolve(generator=self.generate_policy,evaluator=self.evaluate_policy,pop_size=self.pop_size,seed=[],maximize=True,bounder=inspyred.ec.Bounder(xiajie,shangjie),max_generations=self.max_generations,mutation_rate=0.05,num_inputs=step_number*action_dim)

        final_pop.sort(reverse=True)
        self.final_pop = final_pop
        rizhi = '\n\nMXairfoil: (GA) finished, final_pop[0] = '+str(final_pop[0])+'\n converge in '+str(self.converge_history)
        self.jilu(rizhi)
        return final_pop        

    def save_GA(self):
        # output something. result, time consumption,and so on.
        # shijian = time.strftime("%Y-%m-%d-%H:%M", time.localtime())
        location = self.save_location 
        if not(os.path.exists(location)):
            #which means there are no such folder, then mkdir.
            try:
                os.mkdir(location)
            except:
                print('MXairfoil: can not make dir for saveing GA. ',location)
                location = self.location
        
        final_pop_location = location + '/final_pop.pkl' 
        
        try:
            pickle.dump(self.final_pop,open(final_pop_location,'wb'))
        except:
            print('MXairfoil: GA running, no final pop to save.')

        converge_history_location = location + '/converge_history.pkl' 
        pickle.dump(self.converge_history,open(converge_history_location,'wb'))

        candidates_history_location = location + '/candidates.pkl' 
        pickle.dump(self.candidates,open(candidates_history_location,'wb'))

        fitness_location = location + '/fitness.pkl' 
        pickle.dump(self.fitness,open(fitness_location,'wb'))

    def huatu_converge(self):
        tu = huatu(self.converge_history)
        tu.set_location(self.save_location)
        tu.huatu2D('Steps Number','Best Pop','GA Converge history')
        tu.save_all()

    def jilu(self,strBuffer):
        shijian = time.strftime("%Y-%m-%d-%H:%M", time.localtime()) 

        wenjianming = self.save_location + '/log.txt'
        rizhi = open(wenjianming,'a')
        rizhi.write(strBuffer)
        rizhi.write('\n'+shijian+'\n')
        rizhi.close()
        print(strBuffer)
        return

    def record_converge_history(self,N_env,fitness ):
        # chicun = self.converge_history.shape
        self.converge_history = np.append(self.converge_history,np.array([N_env+self.converge_history_last,fitness]).reshape(1,2),axis=0)

    def load_GA(self,**kargs):
        
        if 'location' in kargs:
             self.save_location = kargs['location']

        location = self.save_location

        final_pop_location = location + '/final_pop.pkl' 
        self.final_pop = pickle.load(open(final_pop_location,'rb'))


        converge_history_location = location + '/converge_history.pkl' 
        self.converge_history = pickle.load(open(converge_history_location,'rb'))

        candidates_history_location = location + '/candidates.pkl' 
        self.candidates = pickle.load(open(candidates_history_location,'rb'))

        fitness_location = location + '/fitness.pkl' 
        self.fitness = pickle.load(open(fitness_location,'rb'))

    def post_2D_data(self):
        # this is to generate something that can compare with the lujing.
        # this is for certain policy, 2 dim, 100 steps, CDA42 
        # it seems 4D case can use this too.
        step_number = self.step_number 
        dim = self.action_dim
        pop_size=self.pop_size
        max_generations=self.max_generations
        fitness = self.fitness
        env = self.environment
        try:
            self.final_pop.sort(reverse=True)
            cs = self.final_pop[0].candidate
        except:
            print('MXairfoil: no prepared final_pop. Using candidate and fitness instead')
            max_index = np.argmax(self.fitness)
            cs = self.candidates[max_index]
        state = env.reset_original() # start from original point.
        # cs includes all actions
        total_reward = 0 
        lujing = np.array([]).reshape(0,7)
        
        for j in range(step_number):
            # lujing = np.append(lujing,translate_surrogate(state*1).reshape(1,7),axis=0)
            lujing = np.append(lujing,self.transfer.normal_to_surrogate(state*1).reshape(1,7),axis=0)
            action = cs[dim*j : dim*(j+1)]
            state,reward,done,_ = env.step(action)
            total_reward += reward
            self.N_call_env = self.N_call_env +1 
            if self.N_call_env%1000 == 999:
                print('MXairfoil: calculating, N_call_env = ' +str(self.N_call_env))
        wenjianming_lujing = self.save_location + '/lujing' + str(0) + '.pkl'
        pickle.dump(lujing,open(wenjianming_lujing,'wb'))

    def post_2D_huatu(self):
        # then huatu.
        tu = huatu(0)
        tu.set_location(self.save_location)
        tu.visual_2D(0,0)

if __name__ == '__main__':
    # ENV_NAME = 'shishi_env-v0'
    total_time_start = time.time()
    # ENV_NAME = 'CDA42_env-v0'
    ENV_NAME = 'CDA44_env-v0'
    env = gym.make(ENV_NAME)
    flag = 3
    # 1 for optimized point, 2 for search policy, 3 for post process.
    if flag ==0 :
        print('MXairfoil: nothing happend')
        shishi = GAoptimizer(ENV_NAME)
    elif flag ==1:
        shishi = GAoptimizer(ENV_NAME)
        final_pop = shishi.train_env()
        shishi.save_GA()
        shishi.huatu_converge()
    elif flag ==2:
        # this is for 2D 
        shishi = GAoptimizer(ENV_NAME)
        final_pop = shishi.train_policy()
        shishi.save_GA()
        shishi.huatu_converge()
        shishi.post_2D_data()
        # shishi.post_2D_huatu()
    elif flag ==3:
        # this is to load and do some post process. for 2D policy case.
        shishi = GAoptimizer(ENV_NAME)
        shishi.load_GA()
        # shishi.huatu_converge()
        shishi.post_2D_data()
        shishi.post_2D_huatu()
    elif flag == 4:
        # this is for 4D calculation.
        shishi = GAoptimizer(ENV_NAME)
        final_pop = shishi.train_policy()
        shishi.save_GA()

    total_time_end = time.time()
    total_time_cost = total_time_end - total_time_start
    print('MXairfoil: total time cost ='+str(total_time_cost))
    print('MXairfoil: end a GA optimizer test process, En Taro XXH')
