import numpy as np
from scipy.integrate import odeint
import pandas as pd
from numba.experimental import jitclass
from numba import types, jit
from torch import binomial


class EpiModel(object):
    # variable types
    n_comp: int
    S: int
    E: int
    I_presym: int
    I_asym: int
    I_mild: int
    I_sev: int
    I_hosp: int
    I_icu: int
    R: int
    D: int
    I_hosp_new: int
    I_icu_new: int
    D_new: int
    q_asym: float
    q_sym: float
    gamma: float
    theta: float
    p: np.ndarray
    delta1: float
    delta2: np.ndarray
    delta3: np.ndarray
    delta4: np.ndarray
    psi: np.ndarray
    omega: np.ndarray
    phi1: np.ndarray
    tau1: np.ndarray
    tau2: np.ndarray
    init_state: np.ndarray
    current_state: np.ndarray


    compartments = ['S', 'E', 'I_presym', 'I_asym', 'I_mild', 'I_sev', 'I_hosp', 'I_icu', 'R', 'D', 'I_hosp_new', 'I_icu_new', 'D_new']
    # provide indexes for compartments: self.S = 0, self.E = 1, etc
    for i, name in enumerate(compartments):
        vars()[name] = i
    compartments = compartments
    n_comp = len(compartments)

    @classmethod
    def from_config(cls, config):
        all_cases = pd.read_csv(config['cases'])
        # compute the frequency of confirmed cases for each age group,
        # based on case-data
        # !! for some reason, the rel_freq and n0 where computed using 1-day difference in original paper-code
        cases = all_cases[all_cases['DATE'] >= '2020-03-01']
        cases = cases[cases['DATE'] < '2020-03-14']
        age_cases = cases.groupby('AGEGROUP').agg(np.sum)
        age_cases = age_cases['CASES']
        rel_age_cases = age_cases/age_cases.sum()
        rel_freq = rel_age_cases.values.flatten()

        # n0 is based on age-cases
        cases = all_cases[all_cases['DATE'] >= '2020-03-01']
        cases = cases[cases['DATE'] < '2020-03-13']
        age_cases = cases.groupby('AGEGROUP').agg(np.sum)
        age_cases = age_cases['CASES']
        rel_age_cases = age_cases/age_cases.sum()
        rel_age_cases = rel_age_cases.values
        n0 = age_cases.sum()

        # CREATE S, E
        pop = pd.read_csv(config['population'])
        # [ 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., inf]
        age_groups = np.concatenate((np.arange(0, 100, 10), (np.inf,)))
        g = pd.cut(pop.age, age_groups, right=False)
        pop['group'] = g
        population = pop.groupby('group').agg('sum').population.values.flatten()

        imported_cases = np.round(rel_freq*n0*(1/(1-np.array(config['p']))),0)
        S = population - imported_cases
        E = imported_cases

        return cls(S, E,
                   config['delta2_star'],
                   config['delta3_star'],
                   config['phi0'],
                   config['phi1'],
                   config['mu'],
                   config['q'],
                   config['f'],
                   config['gamma'],
                   config['theta'],
                   config['p'],
                   config['delta1'],
                   config['omega'])

    def __init__(self, S, E,
                 delta2_star,
                 delta3_star,
                 phi0,
                 phi1,
                 mu,
                 q,
                 f,
                 gamma,
                 theta,
                 p,
                 delta1,
                 omega):

        # create init state based on number of compartments and size of S-compartment
        init_state = np.zeros((len(self.compartments), len(S)))
        init_state[self.S] = np.array(S)
        init_state[self.E] = np.array(E)

        self.init_state = init_state
        self.current_state = init_state

        # make numpy array of potential lists
        phi0 = np.array(phi0)
        phi1 = np.array(phi1)
        mu = np.array(mu)
        p = np.array(p)
        omega = np.array(omega)

        self.q_asym = f*q
        self.q_sym = q

        self.gamma = gamma
        self.theta = theta

        self.p = p
        self.delta1 = delta1
	
        self.delta2 = phi0*delta2_star
        self.delta3 = (1-mu)*delta3_star
        self.delta4 = self.delta3
        self.psi = (1-phi0)*delta2_star

        self.omega = omega
        self.phi1 = phi1
       
        self.tau1 = mu*delta3_star
        self.tau2 = self.tau1

    def simulate_day(self, C_asym, C_sym):
        raise NotImplementedError('required for simulation')

    def get_hospitalization_risk(self):
        return self.psi


class ODEModel(EpiModel):

    def deriv(self, y, t, C_asym, C_sym):
        # was flattened for ode
        y = y.reshape(self.init_state.shape)
        d_dt = np.zeros(y.shape)
        # compute lambda
        beta_asym = self.q_asym*C_asym
        beta_sym = self.q_sym*C_sym

        lambda_asym = beta_asym*(y[self.I_presym] + y[self.I_asym])
        lambda_sym = beta_sym*(y[self.I_mild] + y[self.I_sev])
        lambda_ = lambda_asym.sum(1) + lambda_sym.sum(1)

        d_dt[self.S] = -lambda_*y[self.S]
        d_dt[self.E] = lambda_*y[self.S] - self.gamma*y[self.E]
        d_dt[self.I_presym] = self.gamma*y[self.E] - self.theta*y[self.I_presym]
        d_dt[self.I_asym] = self.p*self.theta*y[self.I_presym] - self.delta1*y[self.I_asym]
        d_dt[self.I_mild] = self.theta*(1-self.p)*y[self.I_presym]-(self.psi+self.delta2)*y[self.I_mild]
        d_dt[self.I_sev] = self.psi*y[self.I_mild]-self.omega*y[self.I_sev]
        d_dt[self.I_hosp] = self.phi1*self.omega*y[self.I_sev]-(self.delta3+self.tau1)*y[self.I_hosp]
        d_dt[self.I_icu] = (1-self.phi1)*self.omega*y[self.I_sev]-(self.delta4+self.tau2)*y[self.I_icu]
        d_dt[self.D] = self.tau1*y[self.I_hosp] + self.tau2*y[self.I_icu]
        d_dt[self.R] = self.delta1*y[self.I_asym]+self.delta2*y[self.I_mild]+self.delta3*y[self.I_hosp]+self.delta4*y[self.I_icu]
        # additional compartments for daily hospitalizations, deaths
        d_dt[self.I_hosp_new] = self.phi1*self.omega*y[self.I_sev] - y[self.I_hosp_new]
        d_dt[self.I_icu_new] = (1 - self.phi1)*self.omega*y[self.I_sev] - y[self.I_icu_new]
        d_dt[self.D_new] = self.tau1*y[self.I_hosp] + self.tau1*y[self.I_icu] - y[self.D_new]
        # flatten for ode
        return d_dt.flatten()

    def simulate_day(self, C_asym, C_sym):
        y0 = self.current_state
        # scale of t is "day", so one days passes for each increment of t
        t = np.arange(2, dtype=int)

        ret = odeint(self.deriv, y0.flatten(), t, args=(C_asym, C_sym))
        # state will be the last time period
        self.current_state = ret[-1].reshape(self.init_state.shape)

        return self.current_state

@jit(nopython=True)
def _step_float(n, rate):
    inv_rate = 1-np.exp(rate)
    stepped = np.zeros_like(n)
    for i in range(10):
        stepped[i] = np.random.binomial(n[i], inv_rate)
    return stepped

@jit(nopython=True)
def _step_ndarray(n, rate):
    inv_rate = 1-np.exp(rate)
    stepped = np.zeros_like(n)
    for i in range(10):
        stepped[i] = np.random.binomial(n[i], inv_rate[i])
    return stepped

_step = lambda n, rate: np.random.binomial(n, 1-np.exp(rate))
# _step_ndarray = _step_float = _step

@jit(nopython=True)
def binomial_step(
    n_comp: int,
    S: int,
    E: int,
    I_presym: int,
    I_asym: int,
    I_mild: int,
    I_sev: int,
    I_hosp: int,
    I_icu: int,
    R: int,
    D: int,
    I_hosp_new: int,
    I_icu_new: int,
    D_new: int,
    q_asym: float,
    q_sym: float,
    gamma: float,
    theta: float,
    p: np.ndarray,
    delta1: float,
    delta2: np.ndarray,
    delta3: np.ndarray,
    delta4: np.ndarray,
    psi: np.ndarray,
    omega: np.ndarray,
    phi1: np.ndarray,
    tau1: np.ndarray,
    tau2: np.ndarray,
    y: np.ndarray,
    C_asym: np.ndarray,
    C_sym: np.ndarray,
    h: float,
):
    # modifies `y` in-place
    # compute lambda
    beta_asym = q_asym*C_asym
    beta_sym = q_sym*C_sym

    lambda_asym = beta_asym*(y[I_presym] + y[I_asym])
    lambda_sym = beta_sym*(y[I_mild] + y[I_sev])
    lambda_ = lambda_asym.sum(1) + lambda_sym.sum(1)

    E_n = _step_ndarray(y[S], -h*lambda_)
    I_presym_n = _step_float(y[E], -h*gamma)
    I_asym_n = _step_ndarray(y[I_presym], -h * p * theta)
    I_mild_n = _step_ndarray(y[I_presym], -h * (1-p) * theta)
    I_sev_n = _step_ndarray(y[I_mild], -h * psi)
    I_hosp_n = _step_ndarray(y[I_sev], -h * phi1 * omega)
    I_icu_n = _step_ndarray(y[I_sev], -h * (1-phi1) * omega)
    D_hosp_n = _step_ndarray(y[I_hosp], -h * tau1)
    D_icu_n = _step_ndarray(y[I_icu], -h * tau1)
    R_asym_n = _step_float(y[I_asym], -h * delta1)
    R_mild_n = _step_ndarray(y[I_mild], -h * delta2)
    R_hosp_n = _step_ndarray(y[I_hosp], -h * delta3)
    R_icu_n = _step_ndarray(y[I_icu], -h * delta3)

    y[S] = y[S] - E_n
    y[E] = y[E] + E_n - I_presym_n
    y[I_presym] = y[I_presym] + I_presym_n - I_asym_n - I_mild_n
    y[I_asym] = y[I_asym] + I_asym_n - R_asym_n
    y[I_mild] = y[I_mild] + I_mild_n - I_sev_n - R_mild_n
    y[I_sev] = y[I_sev] + I_sev_n - I_hosp_n - I_icu_n
    y[I_hosp] = y[I_hosp] + I_hosp_n - D_hosp_n - R_hosp_n
    y[I_icu] = y[I_icu] + I_icu_n - D_icu_n - R_icu_n
    y[D] = y[D] + D_hosp_n + D_icu_n
    y[R] = y[R] + R_asym_n + R_mild_n + R_hosp_n + R_icu_n
    # keep track of daily hospitalizations, deaths
    y[I_hosp_new] += I_hosp_n
    y[I_icu_new] += I_icu_n
    y[D_new] += D_hosp_n + D_icu_n

    # clip negative y values (y < 0) to 0
    # y[y<0] = 0
    y.clip(0, None, out=y)


@jit(nopython=True)
def binomial_simulate_day(
    n_comp: int,
    S: int,
    E: int,
    I_presym: int,
    I_asym: int,
    I_mild: int,
    I_sev: int,
    I_hosp: int,
    I_icu: int,
    R: int,
    D: int,
    I_hosp_new: int,
    I_icu_new: int,
    D_new: int,
    q_asym: float,
    q_sym: float,
    gamma: float,
    theta: float,
    p: np.ndarray,
    delta1: float,
    delta2: np.ndarray,
    delta3: np.ndarray,
    delta4: np.ndarray,
    psi: np.ndarray,
    omega: np.ndarray,
    phi1: np.ndarray,
    tau1: np.ndarray,
    tau2: np.ndarray,
    current_state: np.ndarray,
    C_asym: np.ndarray,
    C_sym: np.ndarray,
    h: float,
    h_inv: float,
):
    y = current_state
    # every day reset the 'new' counts
    y[I_hosp_new] = 0.
    y[I_icu_new] = 0.
    y[D_new] = 0.
    # list comprehension is faster than for-loop
    [binomial_step(
        n_comp,
        S,
        E,
        I_presym,
        I_asym,
        I_mild,
        I_sev,
        I_hosp,
        I_icu,
        R,
        D,
        I_hosp_new,
        I_icu_new,
        D_new,
        q_asym,
        q_sym,
        gamma,
        theta,
        p,
        delta1,
        delta2,
        delta3,
        delta4,
        psi,
        omega,
        phi1,
        tau1,
        tau2,
        y,
        C_asym,
        C_sym,
        h) for _ in range(h_inv)]

    return y.copy()


class BinomialModel(EpiModel):

    def __init__(self, *args):
        super(BinomialModel, self).__init__(*args)

        # make states as integers
        self.init_state = self.init_state.astype(int)
        self.current_state = self.current_state.astype(int)

        self.h_inv = 24*10
        self.h = 1/self.h_inv

    def step(self, y, C_asym, C_sym):
        # modifies `y` in-place
        # compute lambda
        beta_asym = self.q_asym*C_asym
        beta_sym = self.q_sym*C_sym

        lambda_asym = beta_asym*(y[self.I_presym] + y[self.I_asym])
        lambda_sym = beta_sym*(y[self.I_mild] + y[self.I_sev])
        lambda_ = lambda_asym.sum(1) + lambda_sym.sum(1)

        E_n = _step(y[self.S], -self.h*lambda_)
        I_presym_n = _step(y[self.E], -self.h*self.gamma)
        I_asym_n = _step(y[self.I_presym], -self.h * self.p * self.theta)
        I_mild_n = _step(y[self.I_presym], -self.h * (1-self.p) * self.theta)
        I_sev_n = _step(y[self.I_mild], -self.h * self.psi)
        I_hosp_n = _step(y[self.I_sev], -self.h * self.phi1 * self.omega)
        I_icu_n = _step(y[self.I_sev], -self.h * (1-self.phi1) * self.omega)
        D_hosp_n = _step(y[self.I_hosp], -self.h * self.tau1)
        D_icu_n = _step(y[self.I_icu], -self.h * self.tau1)
        R_asym_n = _step(y[self.I_asym], -self.h * self.delta1)
        R_mild_n = _step(y[self.I_mild], -self.h * self.delta2)
        R_hosp_n = _step(y[self.I_hosp], -self.h * self.delta3)
        R_icu_n = _step(y[self.I_icu], -self.h * self.delta3)

        y[self.S] = y[self.S] - E_n
        y[self.E] = y[self.E] + E_n - I_presym_n
        y[self.I_presym] = y[self.I_presym] + I_presym_n - I_asym_n - I_mild_n
        y[self.I_asym] = y[self.I_asym] + I_asym_n - R_asym_n
        y[self.I_mild] = y[self.I_mild] + I_mild_n - I_sev_n - R_mild_n
        y[self.I_sev] = y[self.I_sev] + I_sev_n - I_hosp_n - I_icu_n
        y[self.I_hosp] = y[self.I_hosp] + I_hosp_n - D_hosp_n - R_hosp_n
        y[self.I_icu] = y[self.I_icu] + I_icu_n - D_icu_n - R_icu_n
        y[self.D] = y[self.D] + D_hosp_n + D_icu_n
        y[self.R] = y[self.R] + R_asym_n + R_mild_n + R_hosp_n + R_icu_n
        # keep track of daily hospitalizations, deaths
        y[self.I_hosp_new] += I_hosp_n
        y[self.I_icu_new] += I_icu_n
        y[self.D_new] += D_hosp_n + D_icu_n

        # clip negative y values (y < 0) to 0
        y[y<0] = 0

    def simulate_day(self, C_asym, C_sym):
        return binomial_simulate_day(
                self.n_comp,
                self.S,
                self.E,
                self.I_presym,
                self.I_asym,
                self.I_mild,
                self.I_sev,
                self.I_hosp,
                self.I_icu,
                self.R,
                self.D,
                self.I_hosp_new,
                self.I_icu_new,
                self.D_new,
                self.q_asym,
                self.q_sym,
                self.gamma,
                self.theta,
                self.p,
                self.delta1,
                self.delta2,
                self.delta3,
                self.delta4,
                self.psi,
                self.omega,
                self.phi1,
                self.tau1,
                self.tau2,
                self.current_state,
                C_asym,
                C_sym,
                self.h,
                self.h_inv,
        )
        
        y = self.current_state
        # every day reset the 'new' counts
        y[self.I_hosp_new] = 0.
        y[self.I_icu_new] = 0.
        y[self.D_new] = 0.
        # list comprehension is faster than for-loop
        [self.step(y, C_asym, C_sym) for _ in range(self.h_inv)]

        return y.copy()


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='run model')
    parser.add_argument('config', type=str, help='a .json config-file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    contact_types = ['home', 'work', 'transport', 'school', 'leisure', 'otherplace']
    csm = [pd.read_csv(Path(config['social_contact_dir']) / f'{ct}.csv', header=None).values for ct in contact_types]
    csm = np.array(csm)
    beta_0 = config['beta_0']
    beta_1 = config['beta_1']

    model = ODEModel.from_config(config)

    def gradual_compliance_weights(t, beta_0, beta_1):
        x = beta_0 + beta_1*t
        w1 = np.minimum(1, np.exp(x)/(1+np.exp(x)))
        w0 = 1-w1
        return w0, w1

    states = [model.init_state]
    start_lockdown = 14
    for day in range(0, 120):
        # before lockdown
        if day < start_lockdown:
            p_w = p_s = p_l = 1.
            w0, w1 = 1, 0
        else:
            p_w, p_s, p_l = 0.2, 0.0, 0.1
            w0, w1 = gradual_compliance_weights(day-start_lockdown, beta_0, beta_1)
            
        C_sym_factor = np.array([1., 0.09, 0.13, 0.09, 0.06, 0.25])[:, None, None]
        p = np.array([1, p_w, p_w, p_s, p_l, p_l])[:, None, None]
        C_target = csm*p
        C = csm*w0 + C_target*w1
        c_asym = C.sum(0)
        c_sym = (C*C_sym_factor).sum(0)

        next_state = model.simulate_day(c_asym, c_sym)
        states.append(next_state)
    states = np.array(states)
    
    i_hosp_new = states[:, model.I_hosp_new].sum(1)
    i_icu_new = states[:, model.I_icu_new].sum(1)
    d_new = states[:, model.D_new].sum(1)

    plt.figure()
    plt.plot(i_hosp_new, label='hosp')
    plt.plot(i_icu_new, label='icu')
    plt.plot(i_hosp_new+i_icu_new, label='hosp+icu')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(d_new, label='daily death count')
    plt.legend()
    plt.show()