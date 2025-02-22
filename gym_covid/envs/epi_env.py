import numpy as np
import gym
from gym.spaces import Box
import datetime
import pandas as pd


def gradual_compliance_weights(t, beta_0, beta_1):
    x = beta_0 + beta_1*t
    w1 = np.minimum(1, np.exp(x)/(1+np.exp(x)))
    w0 = 1-w1
    return w0, w1


def school_holidays(C, C_current, C_target):
    # schools are closed, 0% school contacts
    C[3] = C[3]*0.
    C_current[3] = C_current[3]*0.
    C_target[3] = C_target[3]*0.
    return C, C_current, C_target

class EpiEnv(gym.Env):

    def __init__(self, model, C=None, beta_0=None, beta_1=None, datapoints=None):
        super(EpiEnv, self).__init__()
        # under the hood we run this epi model
        self.model = model
        # population size of each age-group is sum of the compartments
        N = self.model.init_state.sum(axis=0)
        self.K = len(N)
        self.N = N.sum()
        # contact matrix
        self.C = np.ones((1, self.K, self.K)) if C is None else C
        # factors of contact matrix for symptomatic people, reshape to match C shape
        self.C_sym_factor = np.array([1., 0.09, 0.13, 0.09, 0.06, 0.25])[:, None, None]
        self.C_full = self.C.copy()

        self.beta_0 = beta_0
        self.beta_1 = beta_1

        # the observation space are compartments x age_groups
        self.observation_space = Box(low=np.zeros((model.n_comp, self.K)), high=np.tile(N, (model.n_comp, 1)), dtype=np.float32)
        # action space is proportional reduction of work, school, leisure
        self.action_space = Box(low=np.zeros(3), high=np.ones(3), dtype=np.float32)
        # reward_space is attack-rate for infections, hospitalizations and reduction in social contact
        self.reward_space = Box(low=np.zeros(3), high=np.array([N.sum(), N.sum(), 1]), dtype=np.float32)

        self.datapoints = datapoints
        self.days_per_timestep = 7
        self.today = datetime.date(2020, 3, 1)

        self.events = {}
        # include school holiday
        for holiday_start, holiday_end in ((datetime.date(2020, 7, 1), datetime.date(2020, 8, 31)),
                                           (datetime.date(2020, 11, 2), datetime.date(2020, 11, 8)),
                                           (datetime.date(2020, 12, 21), datetime.date(2021, 1, 3))):
            # enforce holiday-event (0% school contacts) every day of school-holiday
            for i in range((holiday_end-holiday_start).days+1):
                day = holiday_start+datetime.timedelta(days=i)
                self.events[day] = school_holidays

        self.previous_state = None

        self.C_diff_fairness = None



    def reset(self):
        self.model.current_state = self.model.init_state.copy()
        self.current_C = self.C
        self.today = datetime.date(2020, 3, 1)
        # check for events on this day
        event = self.today in self.events
        #print(self.model.current_state)
        return self.model.current_state, event, np.zeros(3)

    def step(self, action):
        # action is a 3d continuous vector
        p_w, p_s, p_l = action
        
        # match all C components, reshape to match C shape
        p = np.array([1, p_w, p_w, p_s, p_l, p_l])[:, None, None]
        C_target = self.C*p

        s = self.model.current_state.copy()
        self.previous_state = s # TODO aangepast

        # simulate for a whole week, sum the daily rewards
        r_ari = r_arh = r_sr = 0.
        state_n = np.empty((self.days_per_timestep,) + self.observation_space.shape)
        event_n = np.zeros((self.days_per_timestep, 1), dtype=bool)

        for day in range(self.days_per_timestep):
            # every day check if there are events on the calendar
            today = self.today + datetime.timedelta(days=day)
            if today in self.events:
                C_full, C_c, C_t = self.events[today](self.C.copy(), self.current_C, C_target)
                #C_full = C_full.sum(0)
                # today is a school holiday
                event_n[day] = True
            else:
                C_full, C_c, C_t = self.C_full, self.current_C, C_target

            # gradual compliance, C_target is only reached after a number of days
            w0, w1 = gradual_compliance_weights(day, self.beta_0, self.beta_1)
            
            C_asym = C_c*w0 + C_t*w1
            #C_asym = C.sum(axis=0)
            C_sym = (C_asym*self.C_sym_factor)#.sum(axis=0)

            s_n = self.model.simulate_day(C_asym.sum(axis=0), C_sym.sum(axis=0))
            state_n[day] = s_n
            # attack rate infected
            S_s = s[self.model.S]
            S_s_n = s_n[self.model.S]
            r_ari += -(np.sum(S_s) - np.sum(S_s_n))
            # attack rate hospitalization
            I_hosp_new_s_n = s_n[self.model.I_hosp_new] + s_n[self.model.I_icu_new]
            r_arh += -np.sum(I_hosp_new_s_n)
            # reduction in social contact
            R_s_n = s_n[self.model.R]
            # all combinations of age groups
            i, j = np.meshgrid(range(self.K), range(self.K))
            C_diff = C_asym-C_full
            # divide by total population to get lost contacts/person, for each social environment
            r_sr += (C_diff*S_s_n[None, i]*S_s_n[None, j] + C_diff*R_s_n[None, i]*R_s_n[None, j]).sum(axis=(1, 2))/self.N

            # update state
            s = s_n

            h = self.model.get_hospitalization_risk()

            S = s[0, :]
            R = s[8, :]

            # C_diff added so it can be accessed for fairness
            self.C_diff_fairness = C_diff

        # update current contact matrix
        self.current_C = C_target
        # update date
        self.today = self.today + datetime.timedelta(days=self.days_per_timestep)
        # social reduction for work, school and leisure
        r_sr_w = r_sr[1]+r_sr[2]
        r_sr_s = r_sr[3]
        r_sr_l = r_sr[4]+r_sr[5]

        # next-state , reward, terminal?, info
        # provide action as proxy for current SCM, impacts progression of epidemic
        return (state_n, event_n, action.copy()), np.array([r_ari, r_arh, r_sr_w, r_sr_s, r_sr_l]), False, {}

    def similarity_metric(self, state1, state2):
        return 1

    def state_to_array(self, state):
        """Used by history to store individuals"""
        # If state is an array, assume it is preprocessed as needed
        return state

    def normalise_state(self, state):
        return state

    def get_all_entities_in_state(self, state, action, true_action, score, reward):
        """Returns a list of combined states, indicating all the entities encountered at timestep t"""
        # print("state:", state)
        # print("action:", action)
        # print("true_action:", true_action)
        # print("score:", score)
        # print("reward:", reward)

        state_df = self.state_df()
        return [(state_df, action, true_action, score, reward)]

    def state_df(self):
        compartments = [
            "S", "E", "I_presym", "I_asym", "I_mild", "I_sev",
            "I_hosp", "I_icu", "R", "D", "I_hosp_new", "I_icu_new", "D_new",
        ]

        age_groups = ["[0, 10[", "[10, 20[", "[20, 30[", "[30, 40[", "[40, 50[", "[50, 60[", "[60, 70[", "[70, 80[",
                      "[80, 90[", "[90, inf["]

        df = pd.DataFrame(data=self.model.current_state, index=compartments).T

        df.index = age_groups

        h = self.model.get_hospitalization_risk()
        df["h_risk"] = h
        df["day"] = self.today

        return df, self.C_diff_fairness
