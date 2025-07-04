# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 21:31:38 2020

@author: Ziyuan Gu
"""

import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint

# =============================================================================
# parameters
# =============================================================================
predict_horizon = 1 #predict horizon (hour)
simulate_horizon = 15 #simulate horizon (hour)
predict_interval = 1/60 #predict interval length (hour)
num_predict_interval = int(predict_horizon/predict_interval) #number of predict intervals
simulate_interval = 1/3600 #simulate interval length (hour)
num_simulate_interval = int(simulate_horizon/simulate_interval) #number of simulate intervals
tau_interval = 1/2 #pricing interval length (hour)
num_tau_interval = int(predict_horizon/tau_interval) #number of predict prices
total_num_tau_interval = int(simulate_horizon/tau_interval) #number of total prices
opt_tau = 0*np.ones(total_num_tau_interval) # prices to be optimized
tau = 0*np.ones(total_num_tau_interval+num_tau_interval-1) #prices to be fixed
lot_cap = 100 #parking lot spaces
curb_cap = 1500 #curb parking spaces
avg_curb_d = 10/1000 #average distance between adjacent curb parking spaces
avg_lot_d = 1.5/1000 #average distance between adjacent lot parking spaces
l_moving = 1 #average moving distance
jam_acc = 2000 #jam accumulation
ff_speed = 45 #free-flow speed
lot_speed = 10 #lot cruising speed
curb_speed = 20 #curb cruising speed
w = .1 #weight of lot cruising time
#demand
rate_1 = 110/60
rate_2 = 180/60
rate_3 = 130/60
rate_4 = 150/60
q_simulate = np.concatenate((np.ones(int(1/simulate_interval))*rate_1*(simulate_interval*3600),
                             np.linspace(rate_1*(simulate_interval*3600), rate_2*(simulate_interval*3600), int(1/simulate_interval)),
                             np.ones(int(1/simulate_interval))*rate_2*(simulate_interval*3600),
                             np.linspace(rate_2*(simulate_interval*3600), rate_1*(simulate_interval*3600), int(1/simulate_interval)),
                             np.ones(int(1/simulate_interval))*rate_1*(simulate_interval*3600),
                             np.linspace(rate_1*(simulate_interval*3600), rate_3*(simulate_interval*3600), int(1/simulate_interval)),
                             np.ones(int(.5/simulate_interval))*rate_3*(simulate_interval*3600),
                             np.linspace(rate_3*(simulate_interval*3600), rate_1*(simulate_interval*3600), int(1/simulate_interval)),
                             np.ones(int(3.5/simulate_interval))*rate_1*(simulate_interval*3600),
                             np.linspace(rate_1*(simulate_interval*3600), rate_4*(simulate_interval*3600), int(1/simulate_interval)),
                             np.ones(int(1/simulate_interval))*rate_4*(simulate_interval*3600),
                             np.linspace(rate_4*(simulate_interval*3600), rate_1*(simulate_interval*3600), int(1/simulate_interval)),
                             np.ones(int(1/simulate_interval))*rate_1*(simulate_interval*3600)), axis=0)
q_predict_all = np.concatenate((np.ones(int(1/predict_interval))*rate_1*(predict_interval*3600),
                             np.linspace(rate_1*(predict_interval*3600), rate_2*(predict_interval*3600), int(1/predict_interval)),
                             np.ones(int(1/predict_interval))*rate_2*(predict_interval*3600),
                             np.linspace(rate_2*(predict_interval*3600), rate_1*(predict_interval*3600), int(1/predict_interval)),
                             np.ones(int(1/predict_interval))*rate_1*(predict_interval*3600),
                             np.linspace(rate_1*(predict_interval*3600), rate_3*(predict_interval*3600), int(1/predict_interval)),
                             np.ones(int(.5/predict_interval))*rate_3*(predict_interval*3600),
                             np.linspace(rate_3*(predict_interval*3600), rate_1*(predict_interval*3600), int(1/predict_interval)),
                             np.ones(int(3.5/predict_interval))*rate_1*(predict_interval*3600),
                             np.linspace(rate_1*(predict_interval*3600), rate_4*(predict_interval*3600), int(1/predict_interval)),
                             np.ones(int(1/predict_interval))*rate_4*(predict_interval*3600),
                             np.linspace(rate_4*(predict_interval*3600), rate_1*(predict_interval*3600), int(1/predict_interval)),
                             np.ones(int(2/predict_interval))*rate_1*(predict_interval*3600)), axis=0)
q_simulate_realized = np.zeros(num_simulate_interval+1)
#logit model coefficients
beta_c_business = -.3
beta_t_business = -.2
beta_c_leisure = -.3
beta_t_leisure = -.1

# =============================================================================
# simulate variables
# =============================================================================
#moving accumulation
moving_curb_acc_s = np.zeros(num_simulate_interval+1)
moving_curb_acc_s[0] = 50
moving_lot_acc_s = np.zeros(num_simulate_interval+1)
moving_lot_acc_s[0] = 0
moving_pass_acc_s = np.zeros(num_simulate_interval+1)
moving_pass_acc_s[0] = 200
moving_acc_s = moving_curb_acc_s+moving_lot_acc_s+moving_pass_acc_s
#cruising accumulation
cruising_acc_s = np.zeros(num_simulate_interval+1)
cruising_acc_s[0] = 0
#total accumulation
acc_s = moving_acc_s+cruising_acc_s
#parking lot accumulation
lot_acc_s = np.zeros(num_simulate_interval+1)
lot_acc_s[0] = 0
#curb parking accumulation
curb_acc_s = np.zeros(num_simulate_interval+1)
curb_acc_s[0] = 500
#tracking time series
ccruising_time_s = np.zeros(num_simulate_interval+1)
ccruising_distance_s = np.zeros(num_simulate_interval+1)
ccruising_time_s_realized = np.zeros(num_simulate_interval+1)
lcruising_time_s = np.zeros(num_simulate_interval+1)
q_l2c_s = np.zeros(num_simulate_interval)
q_curb_s = np.zeros(num_simulate_interval)
q_lot_s = np.zeros(num_simulate_interval)
o_m2c_s = np.zeros(num_simulate_interval)
o_m2l_s = np.zeros(num_simulate_interval)
o_m2p_s = np.zeros(num_simulate_interval)
o_c2c_s = np.zeros(num_simulate_interval)
v_s = np.zeros(num_simulate_interval+1)

# =============================================================================
# predict variables
# =============================================================================
#moving accumulation
moving_curb_acc_p = np.zeros(num_predict_interval+1)
moving_lot_acc_p = np.zeros(num_predict_interval+1)
moving_pass_acc_p = np.zeros(num_predict_interval+1)
moving_acc_p = moving_curb_acc_p+moving_lot_acc_p+moving_pass_acc_p
cruising_acc_p = np.zeros(num_predict_interval+1)
#total accumulation
acc_p = moving_acc_p+cruising_acc_p
#parking lot accumulation
lot_acc_p = np.zeros(num_predict_interval+1)
#curb parking accumulation
curb_acc_p = np.zeros(num_predict_interval+1)
#tracking time series
ccruising_time_p = np.zeros(num_predict_interval+1)
ccruising_distance_p = np.zeros(num_predict_interval+1)
ccruising_time_p_realized = np.zeros(num_predict_interval+1)
lcruising_time_p = np.zeros(num_predict_interval+1)
q_l2c_p = np.zeros(num_predict_interval)
q_curb_p = np.zeros(num_predict_interval)
q_lot_p = np.zeros(num_predict_interval)
o_m2c_p = np.zeros(num_predict_interval)
o_m2l_p = np.zeros(num_predict_interval)
o_m2p_p = np.zeros(num_predict_interval)
o_c2c_p = np.zeros(num_predict_interval)
v_p = np.zeros(num_predict_interval+1)

# =============================================================================
# MPC parameters
# =============================================================================
mpc = True
tau_0 = [i*np.ones(int(predict_horizon/tau_interval)) for i in range(0, 11, 2)]
tau_lower = 0
tau_upper = 10
tau_gap = 5

# =============================================================================
# feedback control parameters
# =============================================================================
feedback = False
occupancy_threshold = 90
i_gain = .2
p_gain = .1
avg_occupancy = np.zeros(total_num_tau_interval)

# =============================================================================
# off-street parking information provision
# =============================================================================
information = False
information_threshold = .1*lot_cap
information_compliance = .5

# =============================================================================
# functions
# =============================================================================
def speedMFD(acc, error=0, ff_speed=ff_speed, jam_acc=jam_acc):
    acc = min(acc, jam_acc-1)
    var = np.random.uniform(1-error*acc/jam_acc, 1+error*acc/jam_acc)
    return (ff_speed-ff_speed*acc/jam_acc)*var

def curbCruisingDistance(curb_acc, acc, a=0.013, b=11.199, curb_speed=curb_speed, d=avg_curb_d, curb_cap=curb_cap):
    if curb_acc/curb_cap <= .5:
        distance = d/curbAvailability(curb_acc)
    else:
        distance = a*np.exp(b*curb_acc/curb_cap)*curb_speed/3600
    return distance
    
def lotCruisingTime(lot_acc, d=avg_lot_d, lot_speed=lot_speed):
    return d/(lotAvailability(lot_acc)*lot_speed) 

def curbAvailability(curb_acc, curb_cap=curb_cap):
    curb_acc = min(curb_acc, curb_cap-1)
    return 1-curb_acc/curb_cap

def lotAvailability(lot_acc, lot_cap=lot_cap):
    lot_acc = min(lot_acc, lot_cap-1)
    return 1-lot_acc/lot_cap

def curbDemand(q_park, beta_c, beta_t, ccruising_t, ctau, ltau):
    return q_park/(1+np.exp(beta_c*(ltau-ctau)-beta_t*ccruising_t*60))

def curbLeave(acc, curb_acc, interval, jam_acc=jam_acc, lower=.3, upper=.3):
    if acc >= jam_acc:
        leave = 0
    else:
        leave = min(curb_acc, jam_acc-acc, interval*3600*np.random.uniform(lower, upper))
    return leave

def lotLeave(acc, lot_acc, curb_leave, interval, jam_acc=jam_acc, lower=.1, upper=.1):
    if acc >= jam_acc:
        leave = 0
    else:
        leave = min(lot_acc, jam_acc-acc-curb_leave, interval*3600*np.random.uniform(lower, upper))
    return leave

def curbCruisingLeave(cruising_acc, curb_acc, interval, curb_cap=curb_cap, threshold=.9, rate=.1, lower=1, upper=1):
    if curb_acc/curb_cap <= threshold:
        leave = 0
    else:
        leave = interval*3600*(rate/(1-threshold))*curb_acc/curb_cap-rate*threshold/(1-threshold)
        leave = leave*np.random.uniform(lower, upper)
    return leave

def ppDemand(q, lower=.2, upper=.2):
    q1 = q*np.random.uniform(lower, upper)
    q2 = q-q1
    return q1, q2

#MPC
def objective(ltau, ctau):
    q_predict_realized = np.zeros(num_predict_interval+1)
    for t in range(1, num_predict_interval+1):
        #departure of parked vehicles
        q_c2m_p = curbLeave(acc_p[t-1], curb_acc_p[t-1], predict_interval)
        q_l2m_p = lotLeave(acc_p[t-1], lot_acc_p[t-1], q_c2m_p, predict_interval)
        #arrival of parking demand
        v_p[t-1] = speedMFD(acc_p[t-1])
        ccruising_distance_p[t-1] = curbCruisingDistance(curb_acc_p[t-1], acc_p[t-1])
        curb_speed_p = min(curb_speed, v_p[t-1])
        ccruising_time_p[t-1] = ccruising_distance_p[t-1]/curb_speed_p
        lcruising_time_p[t-1] = lotCruisingTime(lot_acc_p[t-1])
        #check if jam accumulation is reached and if so, delay arrival of parking demand
        if acc_p[t-1] >= jam_acc:
            q_predict_realized[t] += q_predict_realized[t-1]+q_predict[t-1]
            q_predict_realized[t-1] = 0
        else:
            q_predict_realized[t-1] += q_predict[t-1]
            q_predict_realized[t] += max(0, q_predict_realized[t-1]-(jam_acc-acc_p[t-1]))
            q_predict_realized[t-1] = min(q_predict_realized[t-1], jam_acc-acc_p[t-1])
        q_park_p, q_pass_p = ppDemand(q_predict_realized[t-1])
        q_park_p_business = .2*q_park_p
        q_park_p_leisure = .8*q_park_p
        curb_p_business = curbDemand(q_park_p_business, beta_c_business, beta_t_business, ccruising_time_p[t-1], 
                                     ctau[(t-1)//int(tau_interval/predict_interval)], ltau[(t-1)//int(tau_interval/predict_interval)])
        curb_p_leisure = curbDemand(q_park_p_leisure, beta_c_leisure, beta_t_leisure, ccruising_time_p[t-1], 
                                    ctau[(t-1)//int(tau_interval/predict_interval)], ltau[(t-1)//int(tau_interval/predict_interval)])
        q_curb_p[t-1] = curb_p_business+curb_p_leisure
        q_lot_p[t-1] = q_park_p-q_curb_p[t-1]
        #check if off-street parking information is on
        if information:
            gap = lot_cap-lot_acc_p[t-1]
            if gap <= information_threshold:
                q_curb_p[t-1] = curb_p_business+curb_p_leisure+information_compliance*q_lot_p[t-1]
                q_lot_p[t-1] = q_park_p-q_curb_p[t-1]
        #MFD productions
        cruising_production_p = cruising_acc_p[t-1]*curb_speed_p
        moving_production_p = acc_p[t-1]*v_p[t-1]-cruising_production_p
        #vehicles unable to park off street and cruising on street 
        q_l2c_p[t-1] = max(0, min((moving_lot_acc_p[t-1]/moving_acc_p[t-1])*moving_production_p*predict_interval/l_moving, 
                                  moving_lot_acc_p[t-1]+q_lot_p[t-1])-(lot_cap-lot_acc_p[t-1]+q_l2m_p))
        #on-street cruising vehicles that quit
        q_cbl_p = curbCruisingLeave(cruising_acc_p[t-1], curb_acc_p[t-1], predict_interval)
        #accumulation transfer or completion
        o_m2c_p[t-1] = min((moving_curb_acc_p[t-1]/moving_acc_p[t-1])*moving_production_p*predict_interval/l_moving, 
                           moving_curb_acc_p[t-1]+q_curb_p[t-1])
        o_m2p_p[t-1] = min(.5*(moving_pass_acc_p[t-1]/moving_acc_p[t-1])*moving_production_p*predict_interval/l_moving, 
                           moving_pass_acc_p[t-1]+q_pass_p+q_l2m_p+q_c2m_p+q_cbl_p)
        o_c2c_p[t-1] = min(cruising_production_p*predict_interval/ccruising_distance_p[t-1], 
                           cruising_acc_p[t-1]+q_l2c_p[t-1]-q_cbl_p+o_m2c_p[t-1], 
                           curb_cap-curb_acc_p[t-1]+q_c2m_p)
        o_m2l_p[t-1] = min((moving_lot_acc_p[t-1]/moving_acc_p[t-1])*moving_production_p*predict_interval/l_moving, 
                           moving_lot_acc_p[t-1]+q_lot_p[t-1], lot_cap-lot_acc_p[t-1]+q_l2m_p)
        #accumulation dynamics
        moving_curb_acc_p[t] = moving_curb_acc_p[t-1]+q_curb_p[t-1]-o_m2c_p[t-1]
        moving_lot_acc_p[t] = moving_lot_acc_p[t-1]+q_lot_p[t-1]-q_l2c_p[t-1]-o_m2l_p[t-1]
        moving_pass_acc_p[t] = moving_pass_acc_p[t-1]+q_pass_p+q_l2m_p+q_c2m_p+q_cbl_p-o_m2p_p[t-1]
        moving_acc_p[t] = moving_curb_acc_p[t]+moving_lot_acc_p[t]+moving_pass_acc_p[t]
        cruising_acc_p[t] = cruising_acc_p[t-1]+q_l2c_p[t-1]-q_cbl_p+o_m2c_p[t-1]-o_c2c_p[t-1]
        acc_p[t] = moving_acc_p[t]+cruising_acc_p[t]
        lot_acc_p[t] = lot_acc_p[t-1]-q_l2m_p+o_m2l_p[t-1]
        curb_acc_p[t] = curb_acc_p[t-1]-q_c2m_p+o_c2c_p[t-1]
        #record for the final time point
        if t == num_predict_interval:
            v_p[t] = speedMFD(acc_p[t])
            ccruising_distance_p[t] = curbCruisingDistance(curb_acc_p[t], acc_p[t])
            curb_speed_p = min(curb_speed, v_p[t])
            ccruising_time_p[t] = ccruising_distance_p[t]/curb_speed_p
            lcruising_time_p[t] = lotCruisingTime(lot_acc_p[t])
    #calculate cruising time
    for t in range(num_predict_interval+1):
        ccruising_time_p_realized[t] = min(ccruising_time_p[t], predict_interval)
    total_ccruising_timeseries_p = np.multiply(cruising_acc_p, ccruising_time_p_realized)
    total_forced_timeseries_p = q_l2c_p*lotCruisingTime(lot_cap)
    total_lcruising_timeseries_p = np.multiply(o_m2l_p, lcruising_time_p[1:])
    total_ccruising_time_p = np.sum(total_ccruising_timeseries_p)
    total_forced_time_p = np.sum(total_forced_timeseries_p)
    total_lcruising_time_p = np.sum(total_lcruising_timeseries_p)
    return total_ccruising_time_p+total_forced_time_p+w*total_lcruising_time_p

#plant
for t in range(1, num_simulate_interval+1):
    #check if MPC pricing is on
    if mpc:
        #optimize for every pricing interval
        if (t-1)%int(tau_interval/simulate_interval) == 0:
            moving_curb_acc_p[0] = moving_curb_acc_s[t-1]
            moving_lot_acc_p[0] = moving_lot_acc_s[t-1]
            moving_pass_acc_p[0] = moving_pass_acc_s[t-1]
            moving_acc_p[0] = moving_acc_s[t-1]
            cruising_acc_p[0] = cruising_acc_s[t-1]
            acc_p[0] = acc_s[t-1]
            lot_acc_p[0] = lot_acc_s[t-1]
            curb_acc_p[0] = curb_acc_s[t-1]
            q_predict = q_predict_all[(t-1)//int(predict_interval/simulate_interval):(t-1)//int(predict_interval/simulate_interval)+num_predict_interval]
            tau_previous = opt_tau[max(0, (t-1)//int(tau_interval/simulate_interval)-1)]
            ctau = tau[(t-1)//int(tau_interval/simulate_interval):(t-1)//int(tau_interval/simulate_interval)+num_tau_interval]
            cons = []
            obj = np.inf
            for n in range(len(tau_0)):
                tau_lowers = [tau_lower for i in range(num_tau_interval)]
                tau_uppers = [tau_upper for i in range(num_tau_interval)]
                bnds = Bounds(tau_lowers, tau_uppers)
                def constraint(tau, tau_gap=tau_gap, tau_previous=tau_previous):
                    return [tau_gap-abs(tau[0]-tau_previous), tau_gap-abs(tau[1]-tau[0])]
                nonlinear_con = NonlinearConstraint(constraint, 0, np.inf)
                res = minimize(objective, tau_0[n], args=(ctau), method='trust-constr', bounds=bnds, constraints=nonlinear_con, options={'maxiter': 300})
                if res.fun < obj:
                    obj = res.fun
                    x = res.x[0]
            opt_tau[(t-1)//int(tau_interval/simulate_interval)] = x
    #departure of parked vehicles
    q_c2m_s = curbLeave(acc_s[t-1], curb_acc_s[t-1], simulate_interval, lower=.15, upper=.45)
    q_l2m_s = lotLeave(acc_s[t-1], lot_acc_s[t-1], q_c2m_s, simulate_interval, lower=.05, upper=.15)
    #arrival of parking demand
    v_s[t-1] = speedMFD(acc_s[t-1], error=.2)
    ccruising_distance_s[t-1] = curbCruisingDistance(curb_acc_s[t-1], acc_s[t-1])
    curb_speed_s = min(curb_speed, v_s[t-1])
    ccruising_time_s[t-1] = ccruising_distance_s[t-1]/curb_speed_s
    lcruising_time_s[t-1] = lotCruisingTime(lot_acc_s[t-1])
    #check if jam accumulation is reached and if so, delay arrival of parking demand
    if acc_s[t-1] >= jam_acc:
        q_simulate_realized[t] += q_simulate_realized[t-1]+q_simulate[t-1]
        q_simulate_realized[t-1] = 0
    else:
        q_simulate_realized[t-1] += q_simulate[t-1]
        q_simulate_realized[t] += max(0, q_simulate_realized[t-1]-(jam_acc-acc_s[t-1]))
        q_simulate_realized[t-1] = min(q_simulate_realized[t-1], jam_acc-acc_s[t-1])
    q_park_s, q_pass_s = ppDemand(q_simulate_realized[t-1], lower=.1, upper=.3)
    q_park_s_business = .2*q_park_s
    q_park_s_leisure = .8*q_park_s
    curb_s_business = curbDemand(q_park_s_business, beta_c_business, beta_t_business, ccruising_time_s[t-1], 
                                 tau[(t-1)//int(tau_interval/simulate_interval)], opt_tau[(t-1)//int(tau_interval/simulate_interval)])
    curb_s_leisure = curbDemand(q_park_s_leisure, beta_c_leisure, beta_t_leisure, ccruising_time_s[t-1], 
                                tau[(t-1)//int(tau_interval/simulate_interval)], opt_tau[(t-1)//int(tau_interval/simulate_interval)])
    q_curb_s[t-1] = curb_s_business+curb_s_leisure
    q_lot_s[t-1] = q_park_s-q_curb_s[t-1]
    #check if off-street parking information is on
    if information:
        gap = lot_cap-lot_acc_s[t-1]
        if gap <= information_threshold:
            information_compliance_ = np.random.uniform(information_compliance*0, information_compliance*2)
            q_curb_s[t-1] = curb_s_business+curb_s_leisure+information_compliance_*q_lot_s[t-1]
            q_lot_s[t-1] = q_park_s-q_curb_s[t-1]
    #MFD productions
    cruising_production_s = cruising_acc_s[t-1]*curb_speed_s
    moving_production_s = acc_s[t-1]*v_s[t-1]-cruising_production_s
    #vehicles unable to park off street and cruising on street 
    q_l2c_s[t-1] = max(0, min((moving_lot_acc_s[t-1]/moving_acc_s[t-1])*moving_production_s*simulate_interval/l_moving, 
                              moving_lot_acc_s[t-1]+q_lot_s[t-1])-(lot_cap-lot_acc_s[t-1]+q_l2m_s))
    #on-street cruising vehicles that quit
    q_cbl_s = curbCruisingLeave(cruising_acc_s[t-1], curb_acc_s[t-1], simulate_interval, lower=.5, upper=1.5)
    #accumulation transfer or completion
    o_m2c_s[t-1] = min((moving_curb_acc_s[t-1]/moving_acc_s[t-1])*moving_production_s*simulate_interval/l_moving, 
                       moving_curb_acc_s[t-1]+q_curb_s[t-1])
    o_m2p_s[t-1] = min(.5*(moving_pass_acc_s[t-1]/moving_acc_s[t-1])*moving_production_s*simulate_interval/l_moving, 
                       moving_pass_acc_s[t-1]+q_pass_s+q_l2m_s+q_c2m_s+q_cbl_s)
    o_c2c_s[t-1] = min(cruising_production_s*simulate_interval/ccruising_distance_s[t-1], 
                       cruising_acc_s[t-1]+q_l2c_s[t-1]-q_cbl_s+o_m2c_s[t-1], 
                       curb_cap-curb_acc_s[t-1]+q_c2m_s)
    o_m2l_s[t-1] = min((moving_lot_acc_s[t-1]/moving_acc_s[t-1])*moving_production_s*simulate_interval/l_moving, 
                       moving_lot_acc_s[t-1]+q_lot_s[t-1], lot_cap-lot_acc_s[t-1]+q_l2m_s)
    #accumulation dynamics
    moving_curb_acc_s[t] = moving_curb_acc_s[t-1]+q_curb_s[t-1]-o_m2c_s[t-1]
    moving_lot_acc_s[t] = moving_lot_acc_s[t-1]+q_lot_s[t-1]-q_l2c_s[t-1]-o_m2l_s[t-1]
    moving_pass_acc_s[t] = moving_pass_acc_s[t-1]+q_pass_s+q_l2m_s+q_c2m_s+q_cbl_s-o_m2p_s[t-1]
    moving_acc_s[t] = moving_curb_acc_s[t]+moving_lot_acc_s[t]+moving_pass_acc_s[t]
    cruising_acc_s[t] = cruising_acc_s[t-1]+q_l2c_s[t-1]-q_cbl_s+o_m2c_s[t-1]-o_c2c_s[t-1]
    acc_s[t] = moving_acc_s[t]+cruising_acc_s[t]
    lot_acc_s[t] = lot_acc_s[t-1]-q_l2m_s+o_m2l_s[t-1]
    curb_acc_s[t] = curb_acc_s[t-1]-q_c2m_s+o_c2c_s[t-1]
    #record for the final time point
    if t == num_simulate_interval:
        v_s[t] = speedMFD(acc_s[t], error=.2)
        ccruising_distance_s[t] = curbCruisingDistance(curb_acc_s[t], acc_s[t])
        curb_speed_s = min(curb_speed, v_s[t])
        ccruising_time_s[t] = ccruising_distance_s[t]/curb_speed_s
        lcruising_time_s[t] = lotCruisingTime(lot_acc_s[t])
    #check if feedback control pricing is on 
    if feedback == True:
        if (t-1)%int(tau_interval/simulate_interval) == 0 and (t-1)//int(tau_interval/simulate_interval) != 0:
            avg_occupancy[(t-1)//int(tau_interval/simulate_interval)-1] = np.mean(lot_acc_s[t-int(tau_interval/simulate_interval):t])*100/lot_cap
            if (t-1)//int(tau_interval/simulate_interval) == 1:
                opt_tau[(t-1)//int(tau_interval/simulate_interval)] = min(tau_upper, 
                                                                          max(tau_lower, 
                                                                              opt_tau[(t-1)//int(tau_interval/simulate_interval)-1]+i_gain*(avg_occupancy[(t-1)//int(tau_interval/simulate_interval)-1]-occupancy_threshold)))
            elif (t-1)//int(tau_interval/simulate_interval) > 1:
                opt_tau[(t-1)//int(tau_interval/simulate_interval)] = min(tau_upper, 
                                                                          max(tau_lower, 
                                                                              opt_tau[(t-1)//int(tau_interval/simulate_interval)-1]+i_gain*(avg_occupancy[(t-1)//int(tau_interval/simulate_interval)-1]-occupancy_threshold)+p_gain*(avg_occupancy[(t-1)//int(tau_interval/simulate_interval)-1]-avg_occupancy[(t-1)//int(tau_interval/simulate_interval)-2])))
        if t == num_simulate_interval:
            avg_occupancy[(t-1)//int(tau_interval/simulate_interval)] = np.mean(lot_acc_s[t-int(tau_interval/simulate_interval):t])*100/lot_cap
#calculate cruising time         
for t in range(num_simulate_interval+1):
    ccruising_time_s_realized[t] = min(ccruising_time_s[t], simulate_interval)
total_ccruising_timeseries_s = np.multiply(cruising_acc_s, ccruising_time_s_realized)
total_forced_timeseries_s = q_l2c_s*lotCruisingTime(lot_cap)
total_lcruising_timeseries_s = np.multiply(o_m2l_s, lcruising_time_s[1:])
total_ccruising_time_s = np.sum(total_ccruising_timeseries_s)
total_forced_time_s = np.sum(total_forced_timeseries_s)
total_lcruising_time_s = np.sum(total_lcruising_timeseries_s)
total_time_s = total_ccruising_time_s+total_forced_time_s+w*total_lcruising_time_s
