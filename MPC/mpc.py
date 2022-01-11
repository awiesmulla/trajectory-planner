import numpy as np
#import pytomlpp as toml
import toml
from scipy.optimize import minimize
class State:
    """
    vehicle state class
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, omega=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.omega = omega


class MPC:

    def  __init__(self, filename=None, nx=None, nu=None, constraints=None, cost=None, weights=None, t=5, dt=0.1,
                    max_iter=5, iter_finish=0.1):
        
        if filename is not None:
            config = toml.load(filename)
            self.nx = config['nx']
            self.nu = config['nu']
            self.t = config['t']
            self.dt = config['dt']
            self.dl = config['dl']
            self.weights = config['weights']

            self.max_steer = np.deg2rad(config['max_steer'])
            self.min_steer = np.deg2rad(config['min_steer'])
            self.max_speed = config['max_speed']
            self.min_speed = config['min_speed']
            self.max_acceleration = config['max_acceleration']
            self.min_acceleration = config['min_acceleration']
            self.max_d_steer = np.deg2rad(config['max_d_steer'])
            self.max_curvature = config['max_curvature']
            self.min_curvature = config['min_curvature']

            self.n_ind_search = config['n_ind_search']
            self.max_iter = config['max_iter']
            self.iter_stop = config['iter_stop']

            if config['model']=='bicycle':
                self.cost = self.cost_bicycle
            else:
                self.cost = self.cost_unicycle

        else:
            self.nx = nx
            self.nu = nu
            self.weights = weights
            self.t = t
            self.dt = dt
            self.dl = 1.0
            self.max_iter = max_iter
            self.iter_finish = iter_finish

    
    def solve(self, state, ref, oa, oomega,pind):
        """
        ref is a numpy matrix of
            cx  : course x position list
            cy  : course y position list
            cv  : target velocity
            cyaw: course yaw position list
        """

        # get ref trajectory 
        target_ind = self.calc_nearest_index(state, ref, pind)
        xref, dref = self.calc_ref_trajectory(target_ind, state.v, ref)

        #initial guess i.e. reference states
        if oa is None or oomega is None:
            oa = np.zeros(self.t)
            oomega = np.zeros(self.t)
        
        t = self.t

        x0 = np.concatenate((oa,oomega))
        
        #get constraints and bounds
        constraints,bounds = self.constraints(state, xref)
        x = x0
        #iterative mpc (updaing initial guess at every iteration)
        for i in range(self.max_iter):
            poa, pod = oa[:], oomega[:]
            res = minimize(self.cost, x, args=(state, xref, self), method="SLSQP", bounds=bounds, constraints=constraints)
            if res.success:
                x = res.x
                oa = x[: t]
                oomega = x[t: 2*t]
                du = np.absolute(oa - poa).sum() + np.absolute(oomega - pod).sum()  # calc u change value
                if du <= self.iter_stop:
                    break
            else:
                print('res failed')
        if np.all(np.concatenate((oa,oomega))==x0):
            return None, None, target_ind

        return oa,oomega,target_ind


    def constraints(self, state, xref):
        """
            cons = [{'type':'eq', 'fun': con},
                    {'type':'ineq', 'fun': con_real}]
        """
        t = self.t

        constraints = []
        # constraint for omega of steering angle to be lesser than self.max_d_steer
        # cons = {'type':'ineq', 'fun': lambda x: self.max_d_steer - np.abs(x[t]-state.omega)}
        # constraints.append(cons)

        # constraints for min and max acceleration
        cons = {'type':'ineq', 'fun': lambda x: self.max_acceleration*self.dt - (x[0]-state.v)}
        constraints.append(cons)
        cons = {'type':'ineq', 'fun': lambda x:  ( x[0]-state.v ) - self.min_acceleration*self.dt}
        constraints.append(cons)

        bounds = [(None,None) for _ in range(2*t)]

        for v,omega in zip(range(t), range(t , 2*t)):
            #set upper and lower bounds for velocity and steering angle
            bounds[v] = (self.min_speed,self.max_speed)
            bounds[omega] = (self.min_steer, self.max_steer)
            
            if(omega > t):
                # constraint for omega of steering angle to be lesser than self.max_d_steer
                cons = {'type':'ineq', 'fun': lambda x: self.max_d_steer*self.dt - np.abs(x[omega]-x[omega-1])}
                constraints.append(cons)

                # constraints for min and max acceleration
                cons = {'type':'ineq', 'fun': lambda x: self.max_acceleration*self.dt - (x[v]-x[v-1])}
                constraints.append(cons)
                cons = {'type':'ineq', 'fun': lambda x:  ( x[v]-x[v-1] ) - self.min_acceleration*self.dt}
                constraints.append(cons)

            #constraints for curvature
            # cons = {'type':'ineq', 'fun': lambda x: (self.max_curvature - ( np.abs(np.rad2deg(x[omega])/x[v]))) if x[v]!=0 else -1}
            # cons = {'type':'ineq', 'fun': lambda x: (self.max_curvature - ( np.rad2deg(x[omega])/x[v])) if x[v]!=0 else -1}
            cons = {'type':'ineq', 'fun': lambda x: np.abs(x[v]) - np.abs(np.rad2deg(x[omega])) }
            constraints.append(cons)
            # cons = {'type':'ineq', 'fun': lambda x: ((np.rad2deg(x[omega])/x[v]) - self.min_curvature) if x[v]!=0 else -1}
            # constraints.append(cons)

        
        return constraints,bounds

    @staticmethod
    def cost_bicycle(x, state, xref, mpc):
        t = mpc.t

        v_arr = x[ : t]
        omega_arr = x[t: 2*t] 

        cost=0
        x_arr = np.zeros(mpc.t)
        y_arr = np.zeros(mpc.t)
        yaw_arr = np.zeros(mpc.t)

        x,y,v,yaw = state.x,state.y,state.v,state.yaw
        for t in range(mpc.t):
            x += v*np.cos(yaw)*mpc.dt
            y += v*np.sin(yaw)*mpc.dt
            v = v_arr[t]
            yaw += v*np.tan(omega_arr[t])*mpc.dt
            x_arr[t] = x
            y_arr[t] = y
            yaw_arr[t] = yaw

        a_arr = v_arr[1:]-v_arr[:-1]

        cost += mpc.weights[0] * np.sum((x_arr-xref[0][1:])**2)
        cost += mpc.weights[1] * np.sum((y_arr-xref[1][1:])**2)
        cost += mpc.weights[2] * np.sum((v_arr-xref[2][1:])**2)
        cost += mpc.weights[3] * np.sum((yaw_arr-xref[3][1:])**2)

        cost += mpc.weights[4] * np.sum(a_arr**2)
        cost += mpc.weights[5] * np.sum((omega_arr[1:]-omega_arr[:-1])**2)

        cost += mpc.weights[6] * np.sum((a_arr[1:]-a_arr[:-1])**2)

        return cost

    @staticmethod
    def cost_unicycle(x, state, xref, mpc):
        t = mpc.t

        v_arr = x[ : t]
        omega_arr = x[t: 2*t] 

        cost=0
        x_arr = np.zeros(mpc.t)
        y_arr = np.zeros(mpc.t)
        yaw_arr = np.zeros(mpc.t)
        

        x,y,v,yaw = state.x,state.y,state.v,state.yaw
        for t in range(mpc.t):
            x += v*np.cos(yaw)*mpc.dt
            y += v*np.sin(yaw)*mpc.dt
            v = v_arr[t]
            yaw += omega_arr[t]
            x_arr[t] = x
            y_arr[t] = y
            yaw_arr[t] = yaw

        a_arr = v_arr[1:]-v_arr[:-1]

        cost += mpc.weights[0] * np.sum((x_arr-xref[0][1:])**2)
        cost += mpc.weights[1] * np.sum((y_arr-xref[1][1:])**2)
        cost += mpc.weights[2] * np.sum((v_arr-xref[2][1:])**2)
        cost += mpc.weights[3] * np.sum((yaw_arr-xref[3][1:])**2)
        
        cost += mpc.weights[4] * np.sum(a_arr**2)
        cost += mpc.weights[5] * np.sum((omega_arr[1:]-omega_arr[:-1])**2)
        
        cost += mpc.weights[6] * np.sum((a_arr[1:]-a_arr[:-1])**2)

        return cost

    def calc_ref_trajectory(self, ind, v, ref):
        xref = np.zeros((self.nx, self.t + 1))
        dref = np.zeros((1, self.t + 1))
        ncourse = ref[0].size

        xref[0, 0] = ref[0][ind]
        xref[1, 0] = ref[1][ind]
        xref[2, 0] = ref[2][ind]
        xref[3, 0] = ref[3][ind]
        # dref[0, 0] = 0.0  # steer operational point should be 0

        travel = 0.0

        for i in range(self.t + 1):
            travel += abs(v) * self.dt
            dind = int(round(travel / 1.0))

            if (ind + dind) < ncourse:
                xref[0, i] = ref[0][ind + dind]
                xref[1, i] = ref[1][ind + dind]
                xref[2, i] = ref[2][ind + dind]
                xref[3, i] = ref[3][ind + dind]
            else:
                xref[0, i] = ref[0][ncourse - 1]
                xref[1, i] = ref[1][ncourse - 1]
                xref[2, i] = ref[2][ncourse - 1]
                xref[3, i] = ref[3][ncourse - 1]

        return xref, dref
 
    def calc_nearest_index(self, state, ref, pind):

        dx = np.subtract(state.x,ref[0,pind:(pind + self.n_ind_search)])
        dy = np.subtract(state.y,ref[1,pind:(pind + self.n_ind_search)])

        d = dx**2 + dy**2

        ind = np.argmin(d)
        ind = ind + pind

        return ind

    def calc_cte(self, state, ref, ind):
        return np.sqrt((state.x-ref[0][ind])**2 + (state.y-ref[1][ind])**2)