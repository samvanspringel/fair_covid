import random

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import os

from agent.pcn.main_pcn import multidiscrete_env
from gym_covid.envs import *
import argparse
import torch.nn.functional as F

import wandb
from pytz import timezone

import sys

from scenario.main_pcn_core import ss_emb

sys.path.append("./")  # for command-line execution to find the other packages (e.g. envs)

from fairness import SensitiveAttribute, CombinedSensitiveAttribute
from fairness.fairness_framework import FairnessFramework, ExtendedfMDP
from fairness.group import GroupNotion
from fairness.individual import IndividualNotion
from scenario import FeatureBias
from scenario.fraud_detection.MultiMAuS.simulator import parameters
from scenario.fraud_detection.MultiMAuS.simulator.transaction_model import TransactionModel
from scenario.fraud_detection.env import TransactionModelMDP, FraudFeature
from scenario.job_hiring.features import HiringFeature, Gender, ApplicantGenerator, Nationality
from scenario.job_hiring.env import HiringActions, JobHiringEnv
from scenario.parameter_setup import VSC_SAVE_DIR, device


class MultiDiscreteAction(gym.ActionWrapper):
    def __init__(self, env, action_map):
        super(MultiDiscreteAction, self).__init__(env)
        self.action_map = action_map
        self.action_space = gym.spaces.MultiDiscrete([len(am) for am in action_map])

    def action(self, action):
        return np.array([self.action_map[i][action[i]] for i in range(len(self.action_map))])


class ScaleRewardEnv(gym.RewardWrapper):
    def __init__(self, env, min_=0., scale=1.):
        gym.RewardWrapper.__init__(self, env)
        self.min = min_
        self.scale = scale

    def reward(self, reward):
        return (reward - self.min) / self.scale


class TodayWrapper(gym.Wrapper):
    def __init__(self, env):
        super(TodayWrapper, self).__init__(env)

    def reset(self):
        s = super(TodayWrapper, self).reset()
        ss, se, sa = s[1:] # s TODO changed!
        return (ss[-1].T, se[-1], sa)

    # step function of covid env returns simulation results of every day of timestep
    # only keep current day
    # also discard first reward
    def step(self, action):
        s, r, d, i = super(TodayWrapper, self).step(action)
        ss, se, sa = s[1:] #TODO changed!
        # sum all the social burden objectives together:
        p_tot = r[2:].sum()
        r = np.concatenate((r, p_tot[None]))
        return (ss[-1].T, se[-1], sa), r, d, i


class HistoryEnv(gym.Wrapper):
    def __init__(self, env, size=4):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.size = size
        # will be set in _convert
        self._state = None

        # history stacks observations on dim 0
        low = np.repeat(self.observation_space.low, self.size, axis=0)
        high = np.repeat(self.observation_space.high, self.size, axis=0)
        self.observation_space = gym.spaces.Box(low, high)

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        state = self.env.reset(**kwargs)
        # add history dimension
        s = np.expand_dims(state, 0)
        # fill history with current state
        self._state = np.repeat(s, self.size, axis=0)
        return np.concatenate(self._state, axis=0)

    def step(self, ac):
        state, r, d, i = self.env.step(ac)
        # shift history
        self._state = np.roll(self._state, -1, axis=0)
        # add state to history
        self._state[-1] = state
        return np.concatenate(self._state, axis=0), r, d, i


def create_job_env(args):
    team_size = args.team_size
    episode_length = args.episode_length
    diversity_weight = args.diversity_weight
    # Training environment
    population_file = f'./scenario/job_hiring/data/{args.population}.csv'
    if args.vsc == 0:
        population_file = "." + population_file
    applicant_generator = ApplicantGenerator(csv=population_file, seed=args.seed)

    # Initialise and get features to ignore in distance metrics
    if args.ignore_sensitive:
        exclude_from_distance = (HiringFeature.age, HiringFeature.gender, HiringFeature.nationality,
                                 HiringFeature.married)
    else:
        exclude_from_distance = ()

    # Fairness
    if args.combined_sensitive_attributes == 1:
        sensitive_attribute = CombinedSensitiveAttribute([HiringFeature.gender, HiringFeature.nationality],
                                                         sensitive_values=[Gender.female, Nationality.foreign],
                                                         other_values=[Gender.male, Nationality.belgian])
        inn_sensitive_features = [HiringFeature.gender.value]  # TODO
    elif args.combined_sensitive_attributes == 2:
        sensitive_attribute = [SensitiveAttribute(HiringFeature.gender, sensitive_values=Gender.female,
                                                  other_values=Gender.male),
                               SensitiveAttribute(HiringFeature.nationality, sensitive_values=Nationality.foreign,
                                                  other_values=Nationality.belgian)]
        inn_sensitive_features = [HiringFeature.gender.value, HiringFeature.nationality.value]
    else:
        sensitive_attribute = SensitiveAttribute(HiringFeature.gender, sensitive_values=Gender.female,
                                                 other_values=Gender.male)  # TODO: abstract parameters
        inn_sensitive_features = [HiringFeature.gender.value]

    # No bias
    if args.bias == 0:
        reward_biases = []
    # Bias on gender
    elif args.bias == 1:
        reward_biases = [FeatureBias(features=[HiringFeature.gender], feature_values=[Gender.male], bias=0.1)]
    # Bias on nationality and gender
    elif args.bias == 2:
        reward_biases = [FeatureBias(features=[HiringFeature.gender, HiringFeature.nationality],
                                     feature_values=[Gender.male, Nationality.belgian], bias=0.1)]

    # Create environment
    env = JobHiringEnv(team_size=team_size, seed=args.seed, episode_length=episode_length,  # Required ep length for pcn
                       diversity_weight=diversity_weight, applicant_generator=applicant_generator,
                       reward_biases=reward_biases, exclude_from_distance=exclude_from_distance)

    return env, sensitive_attribute, inn_sensitive_features


def create_fraud_env(args):
    # the parameters for the simulation
    params = parameters.get_default_parameters()  # TODO: abstract parameters
    params['seed'] = args.seed
    params['init_satisfaction'] = 0.9
    params['stay_prob'] = [0.9, 0.6]
    params['num_customers'] = 100
    params['num_fraudsters'] = 10
    # params['end_date'] = datetime(2016, 12, 31).replace(tzinfo=timezone('US/Pacific'))
    # params['end_date'] = datetime(2016, 3, 31).replace(tzinfo=timezone('US/Pacific'))
    # # TODO (90357 yearly) Used for min/max reward
    # episode_length = np.sum(params["trans_per_year"]).astype(int) / 366 * (31 + 29 + 31)
    # 1 week = +- 1728 transactions
    num_transactions = args.n_transactions  # 1000
    params['end_date'] = datetime(2016, 1, 7).replace(tzinfo=timezone('US/Pacific'))
    # episode_length = np.sum(params["trans_per_year"]).astype(int) / 366 * (7)  # (90357) Used for min/max reward
    episode_length = num_transactions
    if args.fraud_proportion != 0:
        curr_sum = np.sum(params['trans_per_year'])
        params['trans_per_year'] = np.array([curr_sum * (1 - args.fraud_proportion),
                                             curr_sum * args.fraud_proportion])

    # Initialise and get features to ignore in distance metrics
    if args.ignore_sensitive:
        exclude_from_distance = (FraudFeature.continent, FraudFeature.country, FraudFeature.card_id)
    else:
        exclude_from_distance = ()

    # TODO: abstract parameters
    # Continents mapping from default parameters: {'EU': 0, 'AS': 1, 'NA': 2, 'AF': 3, 'OC': 4, 'SA': 5}
    # sensitive_attribute = SensitiveAttribute(FraudFeature.continent, sensitive_values=1, other_values=0)
    # NA vs EU instead of AS vs EU to increase population size in both
    if args.combined_sensitive_attributes == 1:
        sensitive_attribute = CombinedSensitiveAttribute([FraudFeature.continent, FraudFeature.merchant_id],
                                                         sensitive_values=[2, 6],
                                                         other_values=[0, None])
        inn_sensitive_features = [FraudFeature.continent.value]  # TODO
    elif args.combined_sensitive_attributes == 2:
        sensitive_attribute = [SensitiveAttribute(FraudFeature.continent, sensitive_values=2,
                                                  other_values=0),
                               SensitiveAttribute(FraudFeature.merchant_id, sensitive_values=6,
                                                  other_values=None)]
        inn_sensitive_features = [FraudFeature.continent.value, FraudFeature.continent.merchant_id]
    else:
        sensitive_attribute = SensitiveAttribute(FraudFeature.continent, sensitive_values=2, other_values=0)
        inn_sensitive_features = [FraudFeature.continent.value]

    # No bias
    if args.bias == 0:
        reward_biases = []
    # Bias on gender
    elif args.bias == 1:
        reward_biases = [FeatureBias(features=[FraudFeature.continent], feature_values=[0], bias=0.1)]
    # Bias on nationality and gender
    elif args.bias == 2:
        reward_biases = [FeatureBias(features=[FraudFeature.continent, FraudFeature.merchant_id],
                                     feature_values=[0, 0], bias=0.1)]  # TODO: which merchant to target

    transaction_model = TransactionModel(params, seed=args.seed)
    env = TransactionModelMDP(transaction_model, do_reward_shaping=True, num_transactions=num_transactions,
                              exclude_from_distance=exclude_from_distance, reward_biases=reward_biases)

    return env, sensitive_attribute, inn_sensitive_features


def create_covid_env(args):
    import torch
    import argparse
    from datetime import datetime
    import uuid
    import os
    import gym_covid


    parser = argparse.ArgumentParser(description='PCN')
    parser.add_argument('--objectives', default=[1, 5], type=int, nargs='+',
                        help='index for ari, arh, pw, ps, pl, ptot')
    parser.add_argument('--env', default='ode', type=str, help='ode or binomial')
    parser.add_argument('--action', default='discrete', type=str, help='discrete, multidiscrete or continuous')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--steps', default=3e5, type=float, help='total timesteps')
    parser.add_argument('--batch', default=256, type=int, help='batch size')
    parser.add_argument('--model-updates', default=50, type=int,
                        help='number of times the model is updated at every training iteration')
    parser.add_argument('--top-episodes', default=200, type=int,
                        help='top-n episodes used to compute target-return and horizon. \
                  Initially fill ER with n random episodes')
    parser.add_argument('--n-episodes', default=10, type=int,
                        help='number of episodes to run between each training iteration')
    parser.add_argument('--er-size', default=400, type=int,
                        help='max size (in episodes) of the ER buffer')
    parser.add_argument('--threshold', default=0.02, type=float, help='crowding distance threshold before penalty')
    parser.add_argument('--noise', default=0.0, type=float, help='noise applied on target-return on batch-update')
    parser.add_argument('--model', default='conv1dsmall', type=str, help='conv1d(big|small), dense(big|small)')
    parser.add_argument('--clip_grad_norm', default=None, type=float, help='clip gradient norm during pcn update')
    args = parser.parse_args()
    print(args)

    device = 'cpu'

    env_type = 'ODE' if args.env == 'ode' else 'Binomial'
    n_evaluations = 1 if env_type == 'ODE' else 10
    scale = np.array([800000, 11000, 50., 20, 50, 120])
    ref_point = np.array([-15000000, -200000, -1000.0, -1000.0, -1000.0, -1000.0]) / scale
    scaling_factor = torch.tensor([[1, 1, 1, 1, 1, 1, 0.1]]).to(device)
    max_return = np.array([0, 0, 0, 0, 0, 0]) / scale
    # max_return = np.array([0, -8000, 0, 0, 0, 0])/scale
    # keep only a selection of objectives

    if args.action == 'discrete':
        env = gym.make(f'BECovidWithLockdown{env_type}Discrete-v0')
        nA = env.action_space.n
    else:
        env = gym.make(f'BECovidWithLockdown{env_type}Continuous-v0')
        if args.action == 'multidiscrete':
            env = multidiscrete_env(env)
            nA = env.action_space.nvec.sum()
        # continuous
        else:
            nA = np.prod(env.action_space.shape)
    env = TodayWrapper(env)
    env = ScaleRewardEnv(env, scale=scale)

    env.nA = nA

    #wandb.init(project='pcn-covid', entity='mreymond', config={k: v for k, v in vars(args).items()})

    # logdir = f'{os.getenv("VSC_SCRATCH", "/tmp")}/pcn/commit_4169d7455fa6f08b4a7fa933d66afb9ae7536ff0/'
    # logdir += '/'.join([f'{k}_{v}' for k, v in vars(args).items()]) + '/'
    # logdir += datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + wandb.run.id + '/'

    print(env)
    return env


def create_fairness_framework_env(args):
    if args.vsc == 1:
        result_dir = VSC_SAVE_DIR
    else:
        result_dir = "/Users/samvanspringel/Documents/School/VUB/Master 2/Jaar/Thesis/Frameworks/fairRL/fairRLresults"

    env_type = args.env
    if args.no_window:
        args.window = None
    is_job_hiring = env_type == "covid"
    is_fraud = False
    # Job hiring
    if is_job_hiring:
        logdir = f"{result_dir}/job_hiring/"
        env, sensitive_attribute, inn_sensitive_features = create_job_env(args)
    # Fraud
    elif is_fraud:
        logdir = f"{result_dir}/fraud_detection/"
        env, sensitive_attribute, inn_sensitive_features = create_fraud_env(args)
    else:
        logdir = f"{result_dir}/covid"
        env = create_covid_env(args)

    #
    #logdir += args.log_dir + "/"
    #logdir += datetime.now().strftime('%Y-%m-%d_%H-%M-%S/')
    #os.makedirs(logdir, exist_ok=True)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    all_group_notions = [GroupNotion.StatisticalParity, GroupNotion.EqualOpportunity,
                         GroupNotion.OverallAccuracyEquality, GroupNotion.PredictiveParity,
                         GroupNotion.StatisticalParity_t, GroupNotion.EqualOpportunity_t,
                         GroupNotion.OverallAccuracyEquality_t, GroupNotion.PredictiveParity_t,
                         ]
    first_ind_f_index = len(all_group_notions) + 1

    if args.single_objective != -1:
        # Only reward and group notions are always kept, single individual notion gets index first_ind_f_index
        args.objectives = [min(args.single_objective, first_ind_f_index)]
        print("Single objective:", args.objectives)
        if args.objectives[0] >= first_ind_f_index:
            args.distance_metrics = args.distance_metrics[:1]
            print("Distance metric:", args.distance_metrics)

    #
    _ind_notions_mapping = {
        first_ind_f_index: IndividualNotion.IndividualFairness,
        first_ind_f_index + 1: IndividualNotion.ConsistencyScoreComplement,
        # first_ind_f_index + 2: IndividualNotion.ConsistencyScoreComplement_INN,  # TODO
        first_ind_f_index + 2: IndividualNotion.IndividualFairness_t,
    }

    all_individual_notions = [_ind_notions_mapping[o] for o in args.objectives if o >= first_ind_f_index]
    if args.no_individual:
        all_individual_notions = []
    elif args.compute_individual:
        all_individual_notions = [IndividualNotion.IndividualFairness, IndividualNotion.ConsistencyScoreComplement,
                                  IndividualNotion.ConsistencyScoreComplement_INN]
        args.distance_metrics = args.distance_metrics[:1] * len(all_individual_notions)
    # TODO: individual fairness notion are calculated as requested: o > 5 will be given an index -1?
    elif first_ind_f_index not in args.objectives:
        # Only #7
        if first_ind_f_index + 1 not in args.objectives:
            o_diff = 2
        # Only 6+
        else:
            o_diff = 1
        args.objectives = [o if o < first_ind_f_index else o - o_diff for o in args.objectives]

    use_discount_history = args.discount_history != 0
    discount_factor = args.discount_factor if use_discount_history else None
    discount_threshold = args.discount_threshold if use_discount_history else None
    fairness_framework = FairnessFramework([a for a in HiringActions], [],
                                           individual_notions=all_individual_notions,
                                           group_notions=all_group_notions,
                                           similarity_metric=env.similarity_metric,
                                           distance_metrics=args.distance_metrics,
                                           alpha=args.fair_alpha,
                                           window=args.window,
                                           discount_factor=discount_factor,
                                           discount_threshold=discount_threshold,
                                           inn_sensitive_features=None,
                                           # inn_sensitive_features=[HiringFeature.gender.value],  # TODO
                                           seed=seed,
                                           steps=int(args.steps),
                                           store_interactions=False, has_individual_fairness=not args.no_individual)

    # Extend the environment with fairness framework
    env = ExtendedfMDP(env, fairness_framework)

    # TODO:  #notions = #group notions + #individual notions with specific similarity distance
    # TODO: max reward still ok with new metrics/group divisions
    _num_group_notions = (len(sensitive_attribute) if args.combined_sensitive_attributes >= 2 else 1) * len(
        all_group_notions)
    _num_notions = _num_group_notions + len(all_individual_notions)
    max_reward = args.episode_length * 1
    scale = np.array([1] + [1] * _num_notions)
    ref_point = np.array([-max_reward] + [-args.episode_length] * _num_notions)
    scaling_factor = torch.tensor([[1.0] + ([1] * _num_notions) + [0.1]]).to(device)
    max_return = np.array([max_reward] + [0] * _num_notions) / scale

    env.nA = env.env.nA
    env.scale = scale

    return env, logdir, ref_point, scaling_factor, max_return


#
fMDP_parser = argparse.ArgumentParser(description='fMDP_parser', add_help=False)
fMDP_parser.add_argument('--objectives', default=[0, 1], type=int, nargs='+',
                         help='index for reward (0), StatisticalParity (1), EqualOpportunity (2), '
                              'OverallAccuracyEquality (3), PredictiveParity (4), '
                              'IndividualFairness (5), ConsistencyScoreComplement (6),'
                              'ConsistencyScoreComplement_INN (7)')
fMDP_parser.add_argument('--single_objective', default=-1, type=int, help="Use a single objective to train on")
fMDP_parser.add_argument('--compute_individual', action='store_true', help='Compute individual fairness, '
                                                                           'regardless of the objectives given for PCN')
fMDP_parser.add_argument('--env', default='job', type=str, help='job or fraud')
#
fMDP_parser.add_argument('--seed', default=0, type=int, help='seed for rng')
fMDP_parser.add_argument('--vsc', default=0, type=int, help='running on local (0) or VSC cluster (1)')
# Job hiring parameters
fMDP_parser.add_argument('--team_size', default=20, type=int, help='maximum team size to reach')
fMDP_parser.add_argument('--episode_length', default=100, type=int, help='maximum episode length')
fMDP_parser.add_argument('--diversity_weight', default=0, type=int, help='diversity weight, complement of skill weight')
fMDP_parser.add_argument('--population', default='belgian_population', type=str,
                         help='the name of the population file')
# Fraud detection parameters
fMDP_parser.add_argument('--n_transactions', default=1000, type=int, help='number of transactions per episode')
fMDP_parser.add_argument('--fraud_proportion', default=0, type=float,
                         help='proportion of fraudulent transactions to genuine. '
                              '0 defaults to default MultiMAuS parameters')
#
fMDP_parser.add_argument('--bias', default=0, type=int, help='Which bias configuration to consider. Default 0: no bias')
fMDP_parser.add_argument('--ignore_sensitive', action='store_true')
# Fairness framework
fMDP_parser.add_argument('--window', default=100, type=int, help='fairness framework window')
fMDP_parser.add_argument('--discount_history', default=0, type=int,
                         help='use a discounted history or sliding window implementation')
fMDP_parser.add_argument('--discount_factor', default=1.0, type=float,
                         help='fairness framework discount factor for history')
fMDP_parser.add_argument('--discount_threshold', default=1e-5, type=float,
                         help='fairness framework discount threshold for history')
fMDP_parser.add_argument('--fair_alpha', default=0.1, type=float, help='fairness framework alpha for similarity metric')
fMDP_parser.add_argument('--wandb', default=1, type=int,
                         help="(Ignored, overrides to 0) use wandb for loggers or save local only")
fMDP_parser.add_argument('--no_window', default=0, type=int, help="Use the full history instead of a window")
fMDP_parser.add_argument('--no_individual', default=0, type=int, help="No individual fairness notions")
fMDP_parser.add_argument('--distance_metrics', default=[], type=str, nargs='+',
                         help='The distance metric to use for every individual fairness notion specified')
#
fMDP_parser.add_argument('--combined_sensitive_attributes', default=0, type=int,
                         help='Use a combination of sensitive attributes to compute fairness notions')
#
fMDP_parser.add_argument('--log_dir', default='new_experiment', type=str, help="Directory where to store results")
fMDP_parser.add_argument('--log_compact', action='store_true', help='Save compact logs to save space.')
