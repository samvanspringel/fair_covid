import random

import torch
import datetime

from agent.pcn.main_pcn import multidiscrete_env
from gym_covid import *
import argparse

from pytz import timezone

import sys
import os

sys.path.append("./")  # for command-line execution to find the other packages (e.g. envs)

from fairness import SensitiveAttribute, CombinedSensitiveAttribute
from fairness.fairness_framework import FairnessFramework, ExtendedfMDP
from fairness.group import GroupNotion, ALL_GROUP_NOTIONS
from fairness.individual import IndividualNotion, ALL_INDIVIDUAL_NOTIONS
from scenario import FeatureBias
from scenario.job_hiring.features import HiringFeature, Gender, ApplicantGenerator, Nationality
from scenario.job_hiring.env import HiringActions, JobHiringEnv
from scenario.parameter_setup import VSC_SAVE_DIR, device



#
Reward_ARI = "Reward_ARI"
Reward_ARH = "Reward_ARH"
Reward_SB_W = "Reward_SB_W"
Reward_SB_S = "Reward_SB_S"
Reward_SB_L = "Reward_SB_L"
ALL_REWARDS = [Reward_ARI, Reward_ARH, Reward_SB_W, Reward_SB_S, Reward_SB_L]
#
ALL_OBJECTIVES = ALL_REWARDS + ALL_GROUP_NOTIONS + ALL_INDIVIDUAL_NOTIONS
SORTED_OBJECTIVES = {o: i for i, o in enumerate(ALL_OBJECTIVES)}
#
OBJECTIVES_MAPPING = {
    # Rewards
    "R_ARI": Reward_ARI,
    "R_ARH": Reward_ARH,
    "R_SB_W": Reward_SB_W,
    "R_SB_S": Reward_SB_S,
    "R_SB_L": Reward_SB_L,
    ""
    # Group notions (over history)
    "SP": GroupNotion.StatisticalParity,
    "EO": GroupNotion.EqualOpportunity,
    "OAE": GroupNotion.OverallAccuracyEquality,
    "PP": GroupNotion.PredictiveParity,
    "PE": GroupNotion.PredictiveEquality,
    "EqOdds": GroupNotion.EqualizedOdds,
    "CUAE": GroupNotion.ConditionalUseAccuracyEquality,
    "TE": GroupNotion.TreatmentEquality,
    # Group notions (over timestep)
    "SP_t": GroupNotion.StatisticalParity_t,
    "EO_t": GroupNotion.EqualOpportunity_t,
    "OAE_t": GroupNotion.OverallAccuracyEquality_t,
    "PP_t": GroupNotion.PredictiveParity_t,
    "PE_t": GroupNotion.PredictiveEquality_t,
    "EqOdds_t": GroupNotion.EqualizedOdds_t,
    "CUAE_t": GroupNotion.ConditionalUseAccuracyEquality_t,
    "TE_t": GroupNotion.TreatmentEquality_t,
    # Individual notions (over history)
    "IF": IndividualNotion.IndividualFairness,
    "CSC": IndividualNotion.ConsistencyScoreComplement,
    "CSC_inn": IndividualNotion.ConsistencyScoreComplement_INN,
    # Individual notions (over timestep)
    "IF_t": IndividualNotion.IndividualFairness_t,
    "SBS": IndividualNotion.SocialBurdenScore,
    "ABFTA": IndividualNotion.AgeBasedFairnessThroughUnawareness
    # TODO: include
    # "CSC_t": IndividualNotion.ConsistencyScoreComplement_t,
    # "CSC_inn_t": IndividualNotion.ConsistencyScoreComplement_INN_t,
}
OBJECTIVES_MAPPING_r = {v: k for k, v in OBJECTIVES_MAPPING.items()}
parser_all_objectives = ", ".join([f"{v if isinstance(v, str) else v.name} ({k})"

                                   for k, v in OBJECTIVES_MAPPING.items()])
def get_objective(obj):
    try:
        return GroupNotion[obj]
    except KeyError:
        pass
    try:
        return IndividualNotion[obj]
    except KeyError:
        pass
    return obj


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
        ss, se, sa = s[1:]  # s TODO changed!
        return (ss[-1].T, se[-1], sa)

    # step function of covid env returns simulation results of every day of timestep
    # only keep current day
    # also discard first reward
    def step(self, action):
        s, r, d, i = super(TodayWrapper, self).step(action)
        ss, se, sa = s[1:]  # TODO changed!
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


def create_covid_env(args):
    import torch
    import argparse

    parser = argparse.ArgumentParser(description='PCN')
    parser.add_argument('--objectives', default=[1, 5], type=int, nargs='+',
                        help='index for ari, arh, pw, ps, pl, ptot')
    parser.add_argument('--env', default='ode', type=str, help='ode or binomial')
    parser.add_argument('--action', default='continuous', type=str, help='discrete, multidiscrete or continuous')
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

    scale = np.array([800000, 11000, 50., 20, 50, 120])

    env_type = 'ODE'
    if args.action == 'discrete':
        env = gym.make(f'BECovidWithLockdown{env_type}Discrete-v0')
        nA = env.action_space.n
    else:
        budget = 5
        env = gym.make(f'BECovidWithLockdown{env_type}Budget{budget}Continuous-v0')
        if args.action == 'multidiscrete':
            env = multidiscrete_env(env)
            nA = env.action_space.nvec.sum()
        # continuous
        else:
            nA = np.prod(env.action_space.shape)
    env = TodayWrapper(env)
    env = ScaleRewardEnv(env, scale=scale)

    env.nA = nA

    # wandb.init(project='pcn-covid', entity='mreymond', config={k: v for k, v in vars(args).items()})

    # logdir = f'{os.getenv("VSC_SCRATCH", "/tmp")}/pcn/commit_4169d7455fa6f08b4a7fa933d66afb9ae7536ff0/'
    # logdir += '/'.join([f'{k}_{v}' for k, v in vars(args).items()]) + '/'
    # logdir += datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + wandb.run.id + '/'

    print(env)
    return env


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


def create_fairness_framework_env(args):
    if args.vsc == 1:
        result_dir = VSC_SAVE_DIR
    else:
        result_dir = "/Users/samvanspringel/Documents/School/VUB/Master 2/Jaar/Thesis/fair_covid/fairRLresults"

    env_type = args.env

    if args.no_window:
        args.window = None
    is_covid = env_type == "covid"
    # Job hiring
    if is_covid:
        logdir = f"{result_dir}/covid/"
        env = create_covid_env(args)
    else:
        logdir = f"{result_dir}/job_hiring/"
        env, sensitive_attribute, inn_sensitive_features = create_job_env(args)
    print(env)
    #
    logdir += datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S/')
    os.makedirs(logdir, exist_ok=True)
    print(logdir)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ALL_OBJECTIVES = ALL_REWARDS + ALL_GROUP_NOTIONS + ALL_INDIVIDUAL_NOTIONS
    sort_objectives = {o: i for i, o in enumerate(ALL_OBJECTIVES)}
    # Check for concatenated arguments for objectives and compute objectives
    _sep = ":"

    if _sep in args.objectives:
        args.objectives = args.objectives.split(_sep)
        print("OBJECTIVES:", args.objectives)
    if _sep in args.compute_objectives:
        args.compute_objectives = args.compute_objectives.split(_sep)
        print("CO", args.compute_objectives)
    all_args_objectives = args.objectives + args.compute_objectives

    ordered_objectives = sorted(all_args_objectives,
                                key=lambda o: SORTED_OBJECTIVES[get_objective(OBJECTIVES_MAPPING[o])])
    args.objectives = [i for i, o in enumerate(ordered_objectives) if o in args.objectives]
    # Check for concatenated distance metrics
    ind_notions = [n for n in all_args_objectives if isinstance(get_objective(OBJECTIVES_MAPPING[n]), IndividualNotion)]
    if len(args.distance_metrics) == 1:
        if _sep in args.distance_metrics[0]:
            args.distance_metrics = args.distance_metrics[0].split(_sep)
            dist_metrics = [(n, d) for n, d in zip(ind_notions, args.distance_metrics)]
            dist_metrics = sorted(dist_metrics, key=lambda x: SORTED_OBJECTIVES[get_objective(OBJECTIVES_MAPPING[x[0]])])
            args.distance_metrics = [d for (n, d) in dist_metrics]
        else:
            args.distance_metrics = args.distance_metrics * len(ind_notions)

    mapped_ordered_notions = [OBJECTIVES_MAPPING[n] for n in ordered_objectives]
    all_group_notions = [n for n in mapped_ordered_notions if isinstance(n, GroupNotion)]
    all_individual_notions = [n for n in mapped_ordered_notions if isinstance(n, IndividualNotion)]

    fairness_framework = FairnessFramework([a for a in HiringActions], [],
                                           individual_notions=all_individual_notions,
                                           group_notions=all_group_notions,
                                           similarity_metric=env.similarity_metric,
                                           distance_metrics=args.distance_metrics,
                                           alpha=args.fair_alpha,
                                           window=args.window,
                                           discount_factor=args.discount_factor if args.discount_history else None,
                                           discount_threshold=args.discount_threshold if args.discount_history else None,
                                           discount_delay=args.discount_delay if args.discount_history else None,
                                           nearest_neighbours=args.nearest_neighbours,
                                           inn_sensitive_features=None,
                                           # inn_sensitive_features=[HiringFeature.gender.value],  # TODO
                                           seed=seed,
                                           steps=int(args.steps),
                                           store_interactions=False,
                                           has_individual_fairness=len(all_individual_notions) > 0)

    # Extend the environment with fairness framework
    env = ExtendedfMDP(env, fairness_framework)

    # TODO: max reward still ok with new metrics/group divisions
    _num_group_notions = (len(sensitive_attribute) if args.combined_sensitive_attributes >= 2 else 1) * len(
        all_group_notions)
    _num_notions = _num_group_notions + len(all_individual_notions)
    max_reward = args.episode_length * 1
    scale = np.array([1] + [1] * _num_notions)  # TODO: treatment equality scale+max
    ref_point = np.array([-max_reward] + [-args.episode_length] * _num_notions)
    scaling_factor = torch.tensor([[1.0] + ([1] * _num_notions) + [0.1]]).to(device)
    max_return = np.array([max_reward] + [0] * _num_notions) / scale

    _num_group_notions = (len(sensitive_attribute) if args.combined_sensitive_attributes >= 2 else 1) * len(
        all_group_notions)
    _num_notions = _num_group_notions + len(all_individual_notions)
    max_reward = args.episode_length * 1
    scale = np.array([800000, 10000, 50., 20, 50, 100])
    ref_point = np.array([-15000000, -200000, -1000.0, -1000.0, -1000.0, -1000.0]) / scale
    scaling_factor = torch.tensor([[1, 1, 1, 1, 1, 1, 0.1]]).to(device)
    max_return = np.array([0, 0, 0, 0, 0, 0]) / scale

    env.nA = env.env.nA
    env.scale = scale
    env.action_space = env.env.action_space

    print(all_args_objectives)
    print(ordered_objectives)
    print(max_return)
    #print(len(all_group_notions), len(all_individual_notions))

    env.nA = env.env.nA
    env.scale = scale
    env.action_space = env.env.action_space

    print("Arguments passed to the script environment:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    return env, logdir, ref_point, scaling_factor, max_return


fMDP_parser = argparse.ArgumentParser(description='fMDP_parser', add_help=False)
#
fMDP_parser.add_argument('--objectives', default="R_ARI:R_ARH:R_SB_W:R_SB_S:R_SB_L",
                         type=str, nargs='+', help='Abbreviations of the fairness notions to optimise, one or more of: '
                                                   f'{parser_all_objectives}. Can be supplied as a single string, with'
                                                   f'the arguments separated by a colon, e.g., "R:SP"')
fMDP_parser.add_argument('--compute_objectives', default="SBS:ABFTA",
                         type=str, nargs='*', help='Abbreviations of the fairness notions to compute, '
                                                   f'in addition to the ones being optimised: {parser_all_objectives}'
                                                   f' Can be supplied as a single string, with the arguments separated '
                                                   f'by a colon, e.g., "EO:OAE:PP:IF:CSC"')
#
fMDP_parser.add_argument('--env', default='covid', type=str, help='job or fraud')
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
fMDP_parser.add_argument('--discount_history', action='store_true',
                         help='use a discounted history instead of a sliding window implementation')
fMDP_parser.add_argument('--discount_factor', default=1.0, type=float,
                         help='fairness framework discount factor for history')
fMDP_parser.add_argument('--discount_threshold', default=1e-5, type=float,
                         help='fairness framework discount threshold for history')
fMDP_parser.add_argument('--discount_delay', default=5, type=int,
                         help='the number of timesteps to consider for the fairness notion to not fluctuate more than '
                              'discount_threshold, before deleting earlier timesteps')
fMDP_parser.add_argument('--nearest_neighbours', default=5, type=int,
                         help='the number of neighbours to consider for individual fairness notions based on CSC')
fMDP_parser.add_argument('--fair_alpha', default=0.1, type=float, help='fairness framework alpha for similarity metric')
fMDP_parser.add_argument('--wandb', default=1, type=int,
                         help="(Ignored, overrides to 0) use wandb for loggers or save local only")
fMDP_parser.add_argument('--no_window', default=0, type=int, help="Use the full history instead of a window")
fMDP_parser.add_argument('--no_individual', default=0, type=int, help="No individual fairness notions")
fMDP_parser.add_argument('--distance_metrics', default=['none'], type=str, nargs='*',
                         help='The distance metric to use for every individual fairness notion specified. '
                              'The distance metrics should be supplied for each individual fairness in the objectives, '
                              'then followed by computed objectives. Can be supplied as a single string, with the '
                              'arguments separated by a colon, e.g., "braycurtis:HEOM"')
#
fMDP_parser.add_argument('--combined_sensitive_attributes', default=0, type=int,
                         help='Use a combination of sensitive attributes to compute fairness notions')
#
fMDP_parser.add_argument('--log_dir', default='new_experiment', type=str, help="Directory where to store results")
fMDP_parser.add_argument('--log_compact', action='store_true', help='Save compact logs to save space.')
fMDP_parser.add_argument('--log_coverage_set_only', action='store_true', help='Save only the coverage set logs')
