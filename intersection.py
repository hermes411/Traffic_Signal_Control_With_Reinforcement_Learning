import optparse
import os
import sys
import numpy as np
import traci

# need to include something for highway

class Intersection:
    def __init__(self, n_id, e_id, s_id, w_id, tls_id, junction_id, ns_green_phase, we_green_phase, road_length, cell_length, speed_limit):
        self._n_id = n_id
        self._e_id = e_id
        self._s_id = s_id
        self._w_id = w_id
        self._tls_id = tls_id
        self._junction_id = junction_id
        self._ns_green_phase = ns_green_phase
        self._we_green_phase = we_green_phase
        self._road_length = road_length
        self._cell_length = cell_length
        self._speed_limit = speed_limit
        self._staying_times = {}
        self._sum_of_staying_times = 0
    
    def get_state(self):
        '''
        Retrieve the state of the sumo intersection, which includes position and speed of vehicles and
        the traffic signal state.
        '''
        position_matrix = [[0 for j in range(12)] for i in range(12)]
        speed_matrix = [[0 for j in range(12)] for i in range(12)]

        junction_x, junction_y = traci.junction.getPosition(self._junction_id)

        # populate position and speed matrix for west road of intersection
        offset = self._road_length - junction_x
        for vehID in traci.edge.getLastStepVehicleIDs(self._w_id):
            row = traci.vehicle.getLaneIndex(vehID)
            col = int((traci.vehicle.getPosition(vehID)[0] + offset) // self._cell_length)
            col = min(col, 11)  # account for case that col is 12

            position_matrix[row][col] = 1
            speed_matrix[row][col] = traci.vehicle.getSpeed(vehID) / self._speed_limit
        
        # populate position and speed matrix for north road of intersection
        offset = self._road_length + junction_y
        for vehID in traci.edge.getLastStepVehicleIDs(self._n_id):
            row = traci.vehicle.getLaneIndex(vehID) + 3
            col = int((offset - traci.vehicle.getPosition(vehID)[1]) // self._cell_length)
            col = min(col, 11)  # account for case that col is 12

            position_matrix[row][col] = 1
            speed_matrix[row][col] = traci.vehicle.getSpeed(vehID) / self._speed_limit

        # populate position and speed matrix for east road of intersection
        offset = junction_x
        for vehID in traci.edge.getLastStepVehicleIDs(self._e_id):
            row = traci.vehicle.getLaneIndex(vehID) + 6
            col = int((traci.vehicle.getPosition(vehID)[0] - offset) // self._cell_length)
            col = min(col, 11)  # account for case that col is 12
            
            position_matrix[row][col] = 1
            speed_matrix[row][col] = traci.vehicle.getSpeed(vehID) / self._speed_limit

        # populate position and speed matrix for south road of intersection
        offset = junction_y
        for vehID in traci.edge.getLastStepVehicleIDs(self._s_id):
            row = traci.vehicle.getLaneIndex(vehID) + 9
            col = int((offset - traci.vehicle.getPosition(vehID)[1]) // self._cell_length)
            col = min(col, 11)  # account for case that col is 12

            position_matrix[row][col] = 1
            speed_matrix[row][col] = traci.vehicle.getSpeed(vehID) / self._speed_limit
        
        # generate the light matrix which will contain the traffic signal state
        light_matrix = []
        if traci.trafficlight.getPhase(self._tls_id) == self._ns_green_phase:
            light_matrix = [0, 1]
        else:
            light_matrix = [1, 0]
        
        p = np.array(position_matrix)
        p = p.reshape(1, 12, 12, 1)    # reshape for CNN layer compatibility

        v = np.array(speed_matrix)
        v = v.reshape(1, 12, 12, 1)    # reshape for CNN layer compatibility
        
        l = np.array(light_matrix)
        l = l.reshape(1, 2, 1) # reshape for CNN layer compatability

        return [p, v, l]
    
    def cumultative_staying_time(self):
        '''
        Get the cumultative staying time of all vehicles in the intersection.
        '''
        # set staying time of vehicles who have left the intersection to 0
        roads = [self._n_id, self._e_id, self._s_id, self._w_id]
        vehicles_in_roads = []
        for road in roads:
            vehicles_in_roads += traci.edge.getLastStepVehicleIDs(road)
        for vehID in self._staying_times:
            if vehID not in vehicles_in_roads:
                self._staying_times[vehID] = 0

        cumultative_staying_time = sum(self._staying_times.values())
        self._sum_of_staying_times += cumultative_staying_time
        return cumultative_staying_time

    def update_staying_times(self):
        '''
        Update the staying time of all vehicles in the intersection.
        '''
        roads = [self._n_id, self._e_id, self._s_id, self._w_id]
        for road in roads:
            for vehID in traci.edge.getLastStepVehicleIDs(road):
                if vehID not in self._staying_times:
                    self._staying_times[vehID] = 1
                else:
                    self._staying_times[vehID] += 1
    

    def sum_of_staying_times(self):
        '''
        Return the sum of staying time across the episode.
        '''
        return self._sum_of_staying_times
        
    def reset_staying_time_info(self):
        '''
        Reset the sum and dictionary for staying times in preparation of a new episode.
        '''
        self._sum_of_staying_times = 0
        self._staying_times = {}

class TrafficGenerator:
    def __init__(self, time_steps):
        self._timp_steps = time_steps
        self._num_cars = 0

    def generate_routefile(self, seed):
        '''
        Generate the route file.
        '''

        np.random.seed(seed)    # make tests reproducible

        # demand per second for different destinations
        pW1 = 1. / 10
        pN1 = 1. / 14
        pS1 = 1. / 14
        pN2 = 1. / 17
        pS2 = 1. / 17
        pHN = 1. / 30
        pHS = 1. / 25

        with open('config/routes.rou.xml', 'w') as routes:
            print('''<routes>\n\t<vType id="average_car" vClass="passenger" accel="3" decel="4.5" minGap="2.5" maxSpeed="45" />
            ''', file=routes)

            # routes leading to north highway
            print('    <route id="W1_HN" edges="we1 we2 we3 we4 hwn" />', file=routes)
            print('    <route id="N1_HN" edges="int1ns1 we2 we3 we4 hwn" />', file=routes)
            print('    <route id="S1_HN" edges="int1sn1 we2 we3 we4 hwn" />', file=routes)
            print('    <route id="N2_HN" edges="int2ns1 we3 we4 hwn" />', file=routes)
            print('    <route id="S2_HN" edges="int2sn1 we3 we4 hwn" />', file=routes)

            # routes leading to the south highway
            print('    <route id="W1_HS" edges="we1 we2 we3 se1 hws" />', file=routes)
            print('    <route id="N1_HS" edges="int1ns1 we2 we3 se1 hws" />', file=routes)
            print('    <route id="S1_HS" edges="int1sn1 we2 we3 se1 hws" />', file=routes)
            print('    <route id="N2_HS" edges="int2ns1 we3 se1 hws" />', file=routes)
            print('    <route id="S2_HS" edges="int2sn1 we3 se1 hws" />', file=routes)

            # routes leading to north road of intersection two
            print('    <route id="W1_N2" edges="we1 we2 int2sn2" />', file=routes)
            print('    <route id="N1_N2" edges="int1ns1 we2 int2sn2" />', file=routes)
            print('    <route id="S1_N2" edges="int1sn1 we2 int2sn2" />', file=routes)
            print('    <route id="S2_N2" edges="int2sn1 int2sn2" />', file=routes)
            print('    <route id="HN_N2" edges="-hwn ew1 ew2 int2sn2" />', file=routes)
            print('    <route id="HS_N2" edges="-hws nw1 ew2 int2sn2" />', file=routes)

            # routes leading to south road of intersection two
            print('    <route id="W1_S2" edges="we1 we2 int2ns2" />', file=routes)
            print('    <route id="N1_S2" edges="int1ns1 we2 int2ns2" />', file=routes)
            print('    <route id="S1_S2" edges="int1sn1 we2 int2ns2" />', file=routes)
            print('    <route id="N2_S2" edges="int2ns1 int2ns2" />', file=routes)
            print('    <route id="HN_S2" edges="-hwn ew1 ew2 int2ns2" />', file=routes)
            print('    <route id="HS_S2" edges="-hws nw1 ew2 int2ns2" />', file=routes)

            # routes leading to north road of intersection one
            print('    <route id="W1_N1" edges="we1 int1sn2" />', file=routes)
            print('    <route id="S1_N1" edges="int1sn1 int1sn2" />', file=routes)
            print('    <route id="N2_N1" edges="int2ns1 ew3 int1sn2" />', file=routes)
            print('    <route id="S2_N1" edges="int2sn1 ew3 int1sn2" />', file=routes)
            print('    <route id="HN_N1" edges="-hwn ew1 ew2 ew3 int1sn2" />', file=routes)
            print('    <route id="HS_N1" edges="-hws nw1 ew2 ew3 int1sn2" />', file=routes)

            # routes leading to south road of intersection one
            print('    <route id="W1_S1" edges="we1 int1ns2" />', file=routes)
            print('    <route id="N1_S1" edges="int1ns1 int1ns2" />', file=routes)
            print('    <route id="N2_S1" edges="int2ns1 ew3 int1ns2" />', file=routes)
            print('    <route id="S2_S1" edges="int2sn1 ew3 int1ns2" />', file=routes)
            print('    <route id="HN_S1" edges="-hwn ew1 ew2 ew3 int1ns2" />', file=routes)
            print('    <route id="HS_S1" edges="-hws nw1 ew2 ew3 int1ns2" />', file=routes)

            # routes leading to west road of intersection one
            print('    <route id="N1_W1" edges="int1ns1 ew4" />', file=routes)
            print('    <route id="S1_W1" edges="int1sn1 ew4" />', file=routes)
            print('    <route id="N2_W1" edges="int2ns1 ew3 ew4" />', file=routes)
            print('    <route id="S2_W1" edges="int2sn1 ew3 ew4" />', file=routes)
            print('    <route id="HN_W1" edges="-hwn ew1 ew2 ew3 ew4" />', file=routes)
            print('    <route id="HS_W1" edges="-hws nw1 ew2 ew3 ew4" />', file=routes)
        
            for i in range(self._timp_steps):
                # destination is west road of intersection one
                if np.random.uniform() < pW1:
                    rl = np.random.uniform()
                    if rl < 0.25:  # take a route with a left turn 
                        route_choice = np.random.randint(1, 4)   # choose a random source
                        if route_choice == 1:
                            print('    <vehicle id="S1_W1_%i" type="average_car" route="N1_W1" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)
                        elif route_choice == 2:
                            print('    <vehicle id="S2_W1_%i" type="average_car" route="N2_W1" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)
                        elif route_choice == 3:
                            print('    <vehicle id="HS_W1_%i" type="average_car" route="N2_W1" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)                        
                        self._num_cars += 1
                    else: # take one of the reamining routes
                        route_choice = np.random.randint(1, 4)   # choose a random source
                        if route_choice == 1:
                            print('    <vehicle id="N1_W1_%i" type="average_car" route="N1_W1" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)
                        elif route_choice == 2:
                            print('    <vehicle id="N2_W1_%i" type="average_car" route="N2_W1" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)
                        elif route_choice == 3:
                            print('    <vehicle id="HN_W1_%i" type="average_car" route="N2_W1" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)
                        self._num_cars += 1

                # destination is north road of intersection one
                if np.random.uniform() < pN1:
                    rls = np.random.uniform()
                    if rls < 0.25:  # take a route with a left turn 
                        route_choice = np.random.randint(1, 4)   # choose a random source
                        if route_choice == 1:
                            print('    <vehicle id="W1_N1_%i" type="average_car" route="W1_N1" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)
                        elif route_choice == 2:
                            print('    <vehicle id="S2_N1_%i" type="average_car" route="S2_N1" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)
                        elif route_choice == 3:
                            print('    <vehicle id="HS_N1_%i" type="average_car" route="HS_N1" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)                        
                        self._num_cars += 1
                    elif rls < 0.55: # take one of the remaining routes with a right turn 
                        route_choice = np.random.randint(1, 3)   # choose a random source
                        if route_choice == 1:
                            print('    <vehicle id="N2_N1_%i" type="average_car" route="N2_N1" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)
                        elif route_choice == 2:
                            print('    <vehicle id="HN_N1_%i" type="average_car" route="HN_N1" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)
                        self._num_cars += 1
                    else: # take the remaining route with no turns
                        print('    <vehicle id="S1_N1_%i" type="average_car" route="S1_N1" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)
                        self._num_cars += 1

                # destination is south road of intersection one
                if np.random.uniform() < pS1:
                    rls = np.random.uniform()
                    if rls < 0.25:  # take a route with a left turn 
                        route_choice = np.random.randint(1, 5)   # choose a random source
                        if route_choice == 1:
                            print('    <vehicle id="N2_S1_%i" type="average_car" route="N2_S1" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)
                        elif route_choice == 2:
                            print('    <vehicle id="S2_S1_%i" type="average_car" route="S2_S1" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)
                        elif route_choice == 3:
                            print('    <vehicle id="HS_S1_%i" type="average_car" route="HS_S1" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)                        
                        elif route_choice == 4:
                            print('    <vehicle id="HN_S1_%i" type="average_car" route="HN_S1" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)                        
                        self._num_cars += 1
                    elif rls < 0.55: # take one of the remaining routes with a right turn 
                        print('    <vehicle id="W1_S1_%i" type="average_car" route="W1_S1" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)
                        self._num_cars += 1
                    else: # take the remaining route with no turns
                        print('    <vehicle id="N1_S1_%i" type="average_car" route="N1_S1" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)
                        self._num_cars += 1

                # destination is north road of intersection two
                if np.random.uniform() < pN2:
                    rls = np.random.uniform()
                    if rls < 0.25:  # take a route with a left turn 
                        route_choice = np.random.randint(1, 5)   # choose a random source
                        if route_choice == 1:
                            print('    <vehicle id="W1_N2_%i" type="average_car" route="W1_N2" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)
                        elif route_choice == 2:
                            print('    <vehicle id="N1_N2_%i" type="average_car" route="N1_N2" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)
                        elif route_choice == 3:
                            print('    <vehicle id="S1_N2_%i" type="average_car" route="S1_N2" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)                        
                        elif route_choice == 4:
                            print('    <vehicle id="HS_N2_%i" type="average_car" route="HS_N2" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)                        
                        self._num_cars += 1
                    elif rls < 0.55: # take one of the remaining routes with a right turn 
                        print('    <vehicle id="HN_N2_%i" type="average_car" route="HN_N2" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)
                        self._num_cars += 1
                    else: # take the remaining route with no turns
                        print('    <vehicle id="S2_N2_%i" type="average_car" route="S2_N2" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)
                        self._num_cars += 1

                # destination is south road of intersection two
                if np.random.uniform() < pS2:
                    rls = np.random.uniform()
                    if rls < 0.25:  # take a route with a left turn 
                        route_choice = np.random.randint(1, 4)   # choose a random source
                        if route_choice == 1:
                            print('    <vehicle id="N1_S2_%i" type="average_car" route="N1_S2" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)
                        elif route_choice == 2:
                            print('    <vehicle id="HN_S2_%i" type="average_car" route="HN_S2" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)
                        elif route_choice == 3:
                            print('    <vehicle id="HS_S2_%i" type="average_car" route="HS_S2" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)                        
                        self._num_cars += 1
                    elif rls < 0.55: # take one of the remaining routes with a right turn
                        route_choice = np.random.randint(1, 3)   # choose a random source
                        if route_choice == 1:
                            print('    <vehicle id="W1_S2_%i" type="average_car" route="W1_S2" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)
                        elif route_choice == 2:
                            print('    <vehicle id="S1_S2_%i" type="average_car" route="S1_S2" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)
                        self._num_cars += 1
                    else: # take the remaining route with no turns
                        print('    <vehicle id="N2_S2_%i" type="average_car" route="N2_S2" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)
                        self._num_cars += 1

                # destination is the north highway
                if np.random.uniform() < pHN:
                    rls = np.random.uniform()
                    if rls < 0.25:  # take a route with a left turn not including turn to enter highway
                        route_choice = np.random.randint(1, 3)   # choose a random source
                        if route_choice == 1:
                            print('    <vehicle id="N1_HN_%i" type="average_car" route="N1_HN" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)
                        elif route_choice == 2:
                            print('    <vehicle id="N2_HN_%i" type="average_car" route="N2_HN" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)
                        self._num_cars += 1
                    elif rls < 0.55: # take one of the remaining routes with a right turn
                        route_choice = np.random.randint(1, 3)   # choose a random source
                        if route_choice == 1:
                            print('    <vehicle id="S1_HN_%i" type="average_car" route="S1_HN" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)
                        elif route_choice == 2:
                            print('    <vehicle id="S2_HN_%i" type="average_car" route="S2_HN" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)
                        self._num_cars += 1
                    else: # take the route with only turn being turn into highway
                        print('    <vehicle id="W1_HN_%i" type="average_car" route="W1_HN" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)
                        self._num_cars += 1

                # destination is the south highway
                if np.random.uniform() < pHS:
                    rls = np.random.uniform()
                    if rls < 0.25:  # take a route with a left turn
                        route_choice = np.random.randint(1, 3)   # choose a random source
                        if route_choice == 1:
                            print('    <vehicle id="N1_HS_%i" type="average_car" route="N1_HS" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)
                        elif route_choice == 2:
                            print('    <vehicle id="N2_HS_%i" type="average_car" route="N2_HS" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)
                        self._num_cars += 1
                    elif rls < 0.55: # take one of the remaining routes with a right turn
                        route_choice = np.random.randint(1, 3)   # choose a random source
                        if route_choice == 1:
                            print('    <vehicle id="S1_HS_%i" type="average_car" route="S1_HS" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)
                        elif route_choice == 2:
                            print('    <vehicle id="S2_HS_%i" type="average_car" route="S2_HS" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)
                        self._num_cars += 1
                    else: # take the route with only turn being turn into highway
                        print('    <vehicle id="W1_HS_%i" type="average_car" route="W1_HS" depart="%i" departLane="random" />' % (self._num_cars, i), file=routes)
                        self._num_cars += 1
  
            print('</routes>', file=routes)
        
        self._num_cars = 0

def set_sumo(sumocfg_file_name, time_steps, nogui):
    '''
    Configure the SUMO command for traci to call when starting.
    '''
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")
    
    from sumolib import checkBinary

    if nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    sumo_cmd = [sumoBinary, "-c", os.path.join('config', sumocfg_file_name), "--no-step-log", "true"]

    return sumo_cmd


