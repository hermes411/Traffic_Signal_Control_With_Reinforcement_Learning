from intersection import Intersection, TrafficGenerator, set_sumo
from dqn_agent import DQNAgent
import random
import traci
import timeit

NE_HIGHWAY_ID = 'hwn'
SE_HIGHWAY_ID = 'hws'
TLS_INT1_ID = 'int1'
TLS_INT2_ID = 'int2'
INT1_JUNCTION_ID = 'int1'
INT2_JUNCTION_ID = 'int2'
NS_GREEN_PHASE = 0
WE_GREEN_PHASE = 4
INT1_W = 'we1'
INT1_E = 'ew3'
INT1_N = 'int1ns1'
INT1_S = 'int1sn1'
INT2_W = 'we2'
INT2_E = 'ew2'
INT2_N = 'int2ns1'
INT2_S = 'int2sn1'
INT_SPEED_LIMIT = 15.64

def update_highway_speeds(highway_speeds: dict, highway_id: str) -> None:
    '''Check if a vehicle has just entered the highway. If they did, add the
    speed with the vehID as the key to the dictionary.'''
    for vehID in traci.edge.getLastStepVehicleIDs(highway_id):
            if vehID not in highway_speeds:
                highway_speeds[vehID] = traci.vehicle.getSpeed(vehID)

def average_highway_speed(highway_speeds: dict):
    '''
    Calculate the average highway speed.
    '''
    return sum(list(highway_speeds.values())) / len(list(highway_speeds.values()))

if __name__ == '__main__':
    random.seed(0) # set the seed for reproducible test results

    time_steps = 3600
    episodes = 20

    sumo_cmd = set_sumo('run.sumocfg', time_steps, nogui=True)

    traffic_gen = TrafficGenerator(time_steps)

    int1 = Intersection(n_id=INT1_N, e_id=INT1_E, s_id=INT1_S, w_id=INT1_W, 
                        tls_id=TLS_INT1_ID, junction_id=INT1_JUNCTION_ID, 
                        ns_green_phase=NS_GREEN_PHASE, we_green_phase=WE_GREEN_PHASE, 
                        road_length=500, cell_length=40, speed_limit=15.64)

    agent1 = DQNAgent(discount_rate=0.95, exploration_rate=0.1, learning_rate=0.0002, 
                    memory_capacity=200, action_size=2, batch_size=32, update_rate=0.001)

    try:
        agent1.load_model_weights('model.weights.h5')
        agent1.save_target_model_weights('target_model.weights.h5')
    except:
        pass

    for ep in range(episodes):
        log = open('log.txt', 'a')
        traffic_gen.generate_routefile(ep)
        traci.start(sumo_cmd)
        step = 0
        highway_speeds = {} # key is vehID and value is the speed they entered the highway

        start_time = timeit.default_timer()
        while traci.simulation.getMinExpectedNumber() > 0 and step < time_steps:
            print("step:", step) # TODO

            # observe the intersection state
            int1_state = int1.get_state()

            # get current cumultative waiting time
            staying_time_start = int1.cumultative_staying_time()

            # choose action
            int1_action = agent1.choose_action(int1_state)

            # execute action
            if (int1_action != int1_state[2][0][0][0]):
                # chosen action is the same so keep traffic signal light unchanged
                phase = 0
                if int1_action == 1:
                    phase = NS_GREEN_PHASE
                else:
                    phase = WE_GREEN_PHASE

                for i in range(10):
                    traci.trafficlight.setPhase(TLS_INT1_ID, phase)
                    step += 1
                    traci.simulationStep()
                    update_highway_speeds(highway_speeds, NE_HIGHWAY_ID)
                    update_highway_speeds(highway_speeds, SE_HIGHWAY_ID)
                    int1.update_staying_times()
            else:
                # chosen action is not the same
                # transition phase
                if int1_action == 1:
                    phase = WE_GREEN_PHASE
                else:
                    phase = NS_GREEN_PHASE
                
                for i in range(6): # turn on yellow light for either NS traffic or WE traffic
                    traci.trafficlight.setPhase(TLS_INT1_ID, phase + 1)
                    step += 1
                    traci.simulationStep()
                    update_highway_speeds(highway_speeds, NE_HIGHWAY_ID)
                    update_highway_speeds(highway_speeds, SE_HIGHWAY_ID)
                    int1.update_staying_times()
                for i in range(10): # turn on green light for left turn
                    traci.trafficlight.setPhase(TLS_INT1_ID, phase + 2)
                    step += 1
                    traci.simulationStep()
                    update_highway_speeds(highway_speeds, NE_HIGHWAY_ID)
                    update_highway_speeds(highway_speeds, SE_HIGHWAY_ID)
                    int1.update_staying_times()
                for i in range(6): # turn on yellow light for left turn
                    traci.trafficlight.setPhase(TLS_INT1_ID, phase + 3)
                    step += 1
                    traci.simulationStep()
                    update_highway_speeds(highway_speeds, NE_HIGHWAY_ID)
                    update_highway_speeds(highway_speeds, SE_HIGHWAY_ID)
                    int1.update_staying_times()
                
                # turn on green light for phase transitioning to
                for i in range(10):
                    traci.trafficlight.setPhase(TLS_INT1_ID, (phase + 4) % 8)
                    step += 1
                    traci.simulationStep()
                    update_highway_speeds(highway_speeds, NE_HIGHWAY_ID)
                    update_highway_speeds(highway_speeds, SE_HIGHWAY_ID)
                    int1.update_staying_times()
                
            # observe reward
            reward = staying_time_start - int1.cumultative_staying_time()
            print('reward: ', reward) # TODO

            # get next state
            next_state = int1.get_state()

            # update weights
            agent1.add_experience(int1_state, int1_action, reward, next_state, False)
            agent1.replay_experience()
            agent1.soft_update_target_network()
        
        mem = agent1._replay_memory[-1]
        del agent1._replay_memory[-1]
        agent1._replay_memory.append((mem[0], mem[1], reward, mem[3], True))

        end_time = timeit.default_timer()
        execution_time = end_time - start_time

        log.write('episode: ' + str(ep + 1) + ',  Sum of staying times: ' + str(int1.sum_of_staying_times()) + ', average highway speed: ' + str(average_highway_speed(highway_speeds)) + ', Execution time: ' + str(execution_time) + '\n')
        log.close()

        int1.reset_staying_time_info()

        traci.close(wait=False)
    
    agent1.save_model_weigths('model.weights.h5')
    agent1.save_target_model_weights('target_model.weights.h5')
