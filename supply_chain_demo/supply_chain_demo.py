"""
DOCSTRING
"""
import argparse
import numpy
import os
import scipy.stats
import time

class InventoryManagement:
    """
    DOCSTRING
    """
    def __init__(self):
        MAX_IPHONES = 100
        MAX_MOVE_OF_IPHONES = 5
        IPHONE_PURCHASES_FIRST_LOC = 3
        IPHONE_PURCHASES_SECOND_LOC = 4
        DELIVERIES_FIRST_LOC = 3
        DELIVERIES_SECOND_LOC = 2
        DISCOUNT = 0.9
        IPHONE_CREDIT = 10
        MOVE_IPHONE_COST = 2
        actions = numpy.arange(-MAX_MOVE_OF_IPHONE, MAX_MOVE_OF_IPHONES + 1)
        POISSON_UPPER_BOUND = 11
        # Probability for poisson distribution
        # NOTE:  lambda should be less than 10 for this function
        self.poisson_cache = dict()

    def __call__(self):
        self.policy_iteration()

    def expected_return(self, state, action, state_value, constant_delivered_iphones):
        """
        Arguments:
            state: [# of iphones in first location, # of iphones in second location]
            action: positive if moving iphones from first location to second location,
                negative if moving ipones from second location to first location
            stateValue: state value matrix
        """
        returns = 0.0
        returns -= MOVE_IPHONE_COST * abs(action)
        # go through all possible iphone purchases
        for iphone_purchases_first_loc in range(0, POISSON_UPPER_BOUND):
            for iphone_purchases_second_loc in range(0, POISSON_UPPER_BOUND):
                # moving iphones
                num_of_iphones_first_loc = int(min(state[0] - action, MAX_IPHONES))
                num_of_iphones_second_loc = int(min(state[1] + action, MAX_IPHONES))
                # valid iphone purchases should be less than actual # of iphones
                real_purchase_first_loc = min(
                    num_of_iphones_first_loc, iphone_purchases_first_loc)
                real_purchase_second_loc = min(
                    num_of_iphones_second_loc, iphones_purchases_second_loc)
                # get credits for purchasing
                reward = (real_purchase_first_loc + real_purchase_second_loc) * purchase_CREDIT
                num_of_iphones_first_loc -= real_purchase_first_loc
                num_of_iphones_second_loc -= real_purchase_second_loc
                # probability for current combination of purchase requests
                prob = self.poisson(purchase_request_first_loc, purchase_REQUEST_FIRST_LOC) * \
                    self.poisson(purchase_request_second_loc, purchase_REQUEST_SECOND_LOC)
                if constant_delivered_iphones:
                    # get delivered iphones, those iphones can be used for purchasing tomorrow
                    delivered_iphones_first_loc = DELIVERIES_FIRST_LOC
                    delivered_iphones_second_loc = DELIVERIES_SECOND_LOC
                    num_of_iphones_first_loc = min(
                        num_of_iphones_first_loc + delivered_iphones_first_loc, MAX_IPHONES)
                    num_of_iphones_second_loc = min(
                        num_of_iphones_second_loc + delivered_iphones_second_loc, MAX_IPHONES)
                    returns += prob * (reward + DISCOUNT * \
                        state_value[num_of_iphones_first_loc, num_of_iphones_second_loc])
        return returns

    def poisson(self, n, lam):
        """
        DOCSTRING
        """
        key = n * 10 + lam
        if key not in self.poisson_cache.keys():
            self.poisson_cache[key] = exp(-lam) * pow(lam, n) / factorial(n)
        return self.poisson_cache[key]

    def policy_iteration(self, constant_delivered_iphones=True):
        """
        DOCSTRING
        """
        value = numpy.zeros((MAX_IPHONES + 1, MAX_IPHONES + 1))
        policy = numpy.zeros(value.shape, dtype=numpy.int)
        # policy evaluation
        while True:
            new_value = numpy.copy(value)
            for i in range(MAX_IPHONES + 1):
                for j in range(MAX_IPHONES + 1):
                    new_value[i, j] = expected_return(
                        [i, j], policy[i, j], new_value, constant_delivered_iphones)
            value_change = numpy.abs((new_value - value)).sum()
            print('value change %f' % (value_change))
            value = new_value
            if value_change < 1e-4:
                break
        # policy improvement
        new_policy = numpy.copy(policy)
        for i in range(MAX_IPHONES + 1):
            for j in range(MAX_IPHONES + 1):
                action_returns = []
                for action in actions:
                    if (action >= 0 and i >= action) or (action < 0 and j >= abs(action)):
                        action_returns.append(
                            expected_return([i, j], action, value, constant_delivered_iphones))
                    else:
                        action_returns.append(-float('inf'))
                new_policy[i, j] = actions[numpy.argmax(action_returns)]
        policy_change = (new_policy != policy).sum()
        print('policy changed in %d states' % (policy_change))
        policy = new_policy
        iterations += 1

class JackCarsModel:
    """
    DOCSTRING
    """
    def __init__(self):
        FLOAT_DATA_TYPE = numpy.float32
        pois_a_rent = scipy.stats.poisson(RENT_EXPECT_A)
        pois_a_return = scipy.stats.poisson(RETURN_EXPECT_A)
        pois_b_rent = scipy.stats.poisson(RENT_EXPECT_B)
        pois_b_return = scipy.stats.poisson(RETURN_EXPECT_B)

    def a_rent_prob(n):
        """
        Distribution of car rental in location a.
        """
        return pois_a_rent.pmf(n)

    def a_return_prob(n):
        """
        Distribution of car return in location a.
        """
        return pois_a_return.pmf(n)

    def b_rent_prob(n):
        """
        Distribution of car rent in location b.
        """
        return pois_b_rent.pmf(n)
    
    def b_return_prob(n):
        """
        Distribution of car return in location b.
        """
        return pois_b_return.pmf(n)

class Model:
    """
    DOCSTRING
    """
    def __init__(self, config):
        """
        Arguments:
            config contains the following attribute
            N_a: possible choices of cars number in "a" location,
                e.g. if number of cars can vary from 0~20, then N_a = 21
            N_b: possible choices of cars number in "b" location
            N_move: maximum number of cars that can be moved
                --> total action number = 2*N_move + 1
            rent_price: revenue earned from renting a car
            move_cost: cost to move a car from a to b or from b to a
            gamma: discount factor in MDP
            P_all_filepath: file path of precomputed P_all matrix
            R_all_filepath: file path of precomputed R_all matrix
        """
        self._N_move = config.N_move
        self._N_act = N_act = 2*self._N_move + 1
        self._N_a = N_a = config.N_a
        self._N_b = N_b = config.N_b
        self._max_storage = max(self._N_a, self._N_b) - 1
        self._rent_price = config.rent_price
        self._move_cost = config.move_cost
        self._gamma = config.gamma
        self._P_filepath = config.P_all_filepath
        self._R_filepath = config.R_all_filepath
        # transition matrix associated with triplet (action, state, next_state)
        self._P_all = numpy.zeros((N_act, N_a*N_b, N_a*N_b), dtype=FLOAT_DATA_TYPE)
        # reward table associated with triplet (action, state, next_state)
        self._R_all = numpy.zeros((N_act, N_a*N_b, N_a*N_b), dtype=FLOAT_DATA_TYPE)
        # transition matrix corresponding to certain policy
        self._P_this_policy = numpy.zeros((N_a*N_b, N_a*N_b), dtype=FLOAT_DATA_TYPE)
        # reward table corresponding to certain policy
        self._R_this_policy = numpy.zeros((N_a*N_b, N_a*N_b), dtype=FLOAT_DATA_TYPE)
        # value function of policy estimated currently
        self._V = numpy.zeros((N_a*N_b), dtype=FLOAT_DATA_TYPE)
        # initialize to random action -N_move~+N_move
        self._policy = numpy.random.randint(
            low=-(self._N_move), high=self._N_move, size=(N_a*N_b), dtype=numpy.int8)

    def compute(self, s_a, s_b, s_a_next, s_b_next, act):
        """
        DOCSTRING
        """
        # compute transition probabilty and expected immediate 
        # reward given this state, next state, and action
        act = act - self._N_move # from range 0~N_act to -N_move~+N_move, +: a-->b, -: b-->a
        # cars moved from one location cannot be more than number of cars in that location
        if (act > 0 and act > s_a) or (act < 0 and -1 * act > s_b):
            return 0, 0
        # compute difference between cars number in current state(today) and next state(tommorrow)
        a_diff = s_a_next - (s_a-act)
        b_diff = s_b_next - (s_b+act)
        # maximum number of cars which can be rented, i.e. number of cars in one location after act
        a_max_rent = s_a - act
        b_max_rent = s_b + act
        # cannot surpass maximum storage
        if a_max_rent>self._max_storage or b_max_rent>self._max_storage:
            return 0, 0 
        # go through all possibility from s_a(today) to s_a_next(tommorrow) with act done overnight
        # in location a
        r_a = p_a = 0
        for a_rent in range(a_max_rent,-1,-1):
            a_return = a_rent + a_diff
            # number of cars returned to location a is not allowed to be negative
            if a_return < 0:
                break
            tmp = a_return_prob(a_return) * a_rent_prob(a_rent)
            r_a = r_a + (a_rent*self._rent_price) * tmp
            p_a = p_a + tmp
        # in location b
        r_b = p_b = 0
        for b_rent in range(b_max_rent, -1, -1):
            b_return = b_rent + b_diff
            # number of cars returned to location a is not allowed to be negative
            if b_return < 0:
                break
            tmp = b_return_prob(b_return) * b_rent_prob(b_rent)
            r_b = r_b + (b_rent*self._rent_price) * tmp
            p_b = p_b + tmp
        # compute total expected reward and transition possibility
        r = r_a + r_b - numpy.absolute(act)*self._move_cost
        p = p_a * p_b
        return p, r

    def form_all(self):
        """
        DOCSTRING
        """
        # form P_all and R_all, basic setup
        print('--form P_all and R_all')
        start = time.time()
        tmp_P = numpy.zeros(
            (self._N_act, self._N_a, self._N_b, self._N_a, self._N_b), dtype=FLOAT_DATA_TYPE)
        tmp_R = numpy.zeros(
            (self._N_act, self._N_a, self._N_b, self._N_a, self._N_b), dtype=FLOAT_DATA_TYPE)
        for s_a in xrange(self._N_a):
            start3 = time.time()
            for s_b in xrange(self._N_b):
                for s_a_next in xrange(self._N_a):
                    for s_b_next in xrange(self._N_b):
                        for act in xrange(self._N_act):
                            tmp_P[act, s_a, s_b, s_a_next, s_b_next], \
                                tmp_R[act, s_a, s_b, s_a_next, s_b_next] \
                                = self.compute(s_a, s_b, s_a_next, s_b_next, act)
            end3= time.time()
            print('*** s_a %d one step time: %f' %(s_a, end3-start3))
        end = time.time()
        # N_state is N_a - 1
        self._P_all = numpy.reshape(tmp_P, self._P_all.shape)
        self._R_all = numpy.reshape(tmp_R, self._P_all.shape)
        print('****** elasped time in form_all %f' %(end-start))

    def get_this_policy_PR(self):
        """
        DOCSTRING
        """
        # update P_this_policy and R_this_policy according to 
        # current policy and looking up P_all and R_all
        # from range -N_move~+N_move to 0~N_act
        index = self._policy + self._N_move
        ### can be better, parellel indexing
        for i in range(self._N_a*self._N_b):
            for j in range(self._N_a*self._N_b):
                self._P_this_policy[i, j] = self._P_all[index[i], i, j]
                self._R_this_policy[i, j] = self._R_all[index[i], i, j]

    def init_value_function(self):
        """
        Initialize value function in the beginning of policy evaluation.
        """
        self._V = self._V

    @property
    def policy(self):
        return self._policy

    def take_step(self, tol):
        """
        DOCSTRING
        """
        # initialize value function V
        self.init_value_function()
        # get P_this_policy and R_this_policy
        self.get_this_policy_PR()
        # policy evaluation
        error = 100
        while error > tol:
            V_next = numpy.tile(self._V, (self._N_a * self._N_b, 1))
            new_V = numpy.sum(
                self._P_this_policy * (self._R_this_policy + self._gamma * V_next), axis=1)
            error = numpy.sum(numpy.square(new_V-self._V))
            print('--value function err = %f' %error)
            self._V = new_V
        # greedy policy improvement
        score = numpy.zeros((self._N_act, self._N_a * self._N_b))
        V_next = numpy.tile(self._V, (self._N_a * self._N_b, 1))
        for act in xrange(self._N_act):
            score[act] = numpy.sum(
                self._P_all[act] * (self._R_all[act] + self._gamma * V_next), axis=1)
        new_policy = numpy.argmax(score, axis=0)
        # from range 0~10 to -N_move~+N_move
        new_policy = new_policy - self._N_move
        return new_V, new_policy

    def train(self, tol, to_form_all=False):
        """
        Arguments:
            tol: tolerance of error in policy evaluation
            to_form_all: True-->will form new P_all and R_all matrix
                False-->load precomputed P_all and R_all matrix 
        """
        print('start training')
        if to_form_all:
            self.form_all()
            f_name = 'P_all_%d_%d' %(self._max_storage, self._N_move)
            numpy.save(f_name, self._P_all)
            f_name = 'R_all_%d_%d' %(self._max_storage, self._N_move)
            numpy.save(f_name, self._R_all)
        else:
            self._P_all = numpy.load(self._P_filepath)
            self._R_all = numpy.load(self._R_filepath)
        print('initial policy')
        print(numpy.reshape(self._policy, (self._N_a,self._N_b)))
        error = 100
        n_iters = 0
        while error != 0:
            n_iters = n_iters + 1
            print('Iter%d:' % n_iters)
            # take one step --> policy evaluation and policy improvement
            new_V, new_policy = self.take_step(tol)
            # check difference between current policy and improved policy
            error = numpy.sum(numpy.absolute(new_policy - self._policy))
            # check for every state if value function with improved 
            # policy is better than that with old policy
            tmp = new_V - self._V
            if tmp[tmp < 0].any():
                print('ERROR: value function do not improve for all states.')
                break
            # update policy and value function
            self._policy = new_policy
            self._V = new_V
            print(numpy.reshape(self._policy, (self._N_a,self._N_b)))
            print('--policy err = %d' %error)
        print('end training')

    @property
    def V(self):
        return self._V

class Parameters:
    """
    DOCSTRING
    """
    N_MOVE = 5
    N_A = 21
    N_B = 21
    RENT_PRICE = 10
    MOVE_COST = 2
    GAMMA = 0.9
    RENT_EXPECT_A = 3
    RENT_EXPECT_B = 4
    RETURN_EXPECT_A = 3
    RETURN_EXPECT_B = 2
    DEFAULT_P_PATH = 'dynamics_mat/P_all_20_5.npy'
    DEFAULT_R_PATH = 'dynamics_mat/R_all_20_5.npy'
    POLICY_EVAL_TOL = 1E-8

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--N_move', type=int, default=N_MOVE,
        help='Maximum number of car that can be moved, default 5')
    parser.add_argument(
        '--N_a', type=int, default=N_A,
        help='Choices of cars number in location a, default 21 --> 0~20 cars')
    parser.add_argument(
        '--N_b', type=int, default=N_B,
        help='Choices of cars number in location b, default 21 --> 0~20 cars')
    parser.add_argument(
        '--rent_price', type=int, default=RENT_PRICE,
        help='Money earned from renting a car, default 4')
    parser.add_argument(
        '--move_cost', type=int, default=MOVE_COST,
        help='Cost of moving one car from one place to the other, default 2')
    parser.add_argument(
        '--gamma', type=float, default=GAMMA,
        help='Discount factor of MDP, default 0.9')
    parser.add_argument(
        '--to_form_all', type=bool, default=False,
        help='True if you want new matrices associated with problem setting, default False')
    parser.add_argument(
        '--P_all_filepath', type=str, default=DEFAULT_P_PATH,
        help='File path of precomputed P_all matrix, default dynamics/P_all_20_5.npy')
    parser.add_argument(
        '--R_all_filepath', type=str, default=DEFAULT_R_PATH,
        help='File path of precomputed R_all matrix, default dynamics/R_all_20_5.npy')
    config = parser.parse_args()
    model = JackCarsModel.Model(config)
    model.train(POLICY_EVAL_TOL, to_form_all=config.to_form_all)
    optim_policy = model.policy 
    optim_policy = numpy.reshape(optim_policy, (N_A,N_B))
    print('optimal policy:', optim_policy)

if __name__=='__main__':
    main()