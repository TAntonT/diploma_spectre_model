"""
Module for communication with environment.
Also designed to test the model.
"""

from random import random, choice
import numpy as np


class TestEnvironment:
    """ Save and update all information about environment.
    """

    def __init__(self, primary_arms_proba: list, repeated_arms_proba: list, constraints: list, failure = False):

        self.n_arms = len(primary_arms_proba)

        # test environment parameters
        self.arms_proba = {'primary': primary_arms_proba,
                           'repeated': repeated_arms_proba}
        self.basic_constraints = constraints

        self.failure = failure

        self.n_payments = 0
        self.n_success = 0

        self.n_primary_payments = 0
        self.n_primary_success = 0

        self.n_repeated_payments = 0
        self.n_repeated_success = 0

        self.n_cascade_payments = 0
        self.n_cascade_success = 0

        self.constraints = self.basic_constraints
        self.temp_constraints = {}
        self.temp_iteration = 0

        # real environment parameters
        self.primary_alphas = np.ones(self.n_arms)
        self.primary_betas = np.ones(self.n_arms)

        self.repeated_alphas = np.ones(self.n_arms)
        self.repeated_betas = np.ones(self.n_arms)

        self.cascade_config = []
        self.historical_cascade_config = []

        self.cascade_alphas = {}
        self.cascade_betas = {}
        self.cascade_mean = {}

    def flush(self):
        """Set to zero."""

        # test environment parameters
        self.n_payments = 0
        self.n_success = 0

        self.n_primary_payments = 0
        self.n_primary_success = 0

        self.n_repeated_payments = 0
        self.n_repeated_success = 0

        self.n_cascade_payments = 0
        self.n_cascade_success = 0

        self.constraints = self.basic_constraints
        self.temp_constraints = []

        # real environment parameters
        self.primary_alphas = np.ones(self.n_arms)
        self.primary_betas = np.ones(self.n_arms)

        self.repeated_alphas = np.ones(self.n_arms)
        self.repeated_betas = np.ones(self.n_arms)

        self.cascade_config = []
        self.cascade_alphas = {}
        self.cascade_betas = {}
        self.cascade_mean = {}

    def get_bank_list(self):
        """ Get list of banks for malfunction_generator.
        :return:
        """
        return [s for s in range(self.n_arms) if self.constraints[s] > 41]

    def malfunction_generator(self):
        """ Simulating the failure of the bank.
        :return:
        """
        bank_list = self.get_bank_list()
        if self.n_cascade_payments == 139  and len(bank_list) > 1:
            bank = choice(bank_list)
            return bank, self.n_cascade_payments
        return None, None

    def delete_bank(self):
        """ Delete bank by failure reason.
        :return:
        """
        bank, iteration = self.malfunction_generator()
        if bank is not None:
            self.temp_constraints = {bank: self.constraints[bank]}
            self.temp_iteration = iteration + 40
            self.constraints[bank] = -100
            return 'Bank ' + str(bank) + ' was deleted at ' + str(iteration) + ' iteration!'

        return None

    def add_bank(self):
        """ Return bank after failure.
        :return:
        """
        if self.n_cascade_payments == self.temp_iteration and len(self.temp_constraints) > 0:
            bank, constraint = self.temp_constraints.popitem()
            self.temp_iteration = 0
            self.constraints[bank] = constraint
            return 'Bank ' + str(bank) + ' was returned at ' + str(self.n_cascade_payments) + ' iteration!'

        return None

    def pull_arm(self, probability):
        """ Simulate arm pulling.

        Return feedback from automate {0,1}.
        """

        if random() < probability:
            return 1
        else:
            return 0

    def update_cascade_mean(self, config):
        self.cascade_mean.update({
            config: self.cascade_alphas[config]/(self.cascade_alphas[config]+ self.cascade_betas[config])
        })

    def get_cascade_mean(self):
        """ Return dictionary of sorted cascade means filtered by cascade_list.

        :param cascade_list:
        :return:
        """

        cascade_mean_dict = self.cascade_mean
        cascade_list = self.get_cascade_config()

        filtered_cascade_mean_dict = {k: v for k, v in cascade_mean_dict.items() if k in cascade_list}

        sorted_cascade_list = sorted(filtered_cascade_mean_dict,
                                          key=filtered_cascade_mean_dict.__getitem__,
                                          reverse=True)

        return sorted_cascade_list

    def update_cascade_config(self, config: str):
        if config not in self.cascade_config:
            self.cascade_config.append(config)
            self.historical_cascade_config.append(config)
            self.cascade_alphas.update({config: 1})
            self.cascade_betas.update({config: 1})
            self.update_cascade_mean(config)

    def get_cascade_config(self):
        """ Return cascade config filtered by constraints

        :return:
        """
        print(self.cascade_config)
        cascade_list = self.cascade_config
        for cascade in cascade_list:
            for s in [int(x) for x in cascade.split()]:
                if self.constraints[s] <= 0:
                    cascade_list.remove(cascade)
        print(cascade_list)
        # Check if cascade list is null
        if len(cascade_list) == 0:
            # Exception if cascade is null
            print('Cascade list is empty!!!')
        else:
            return cascade_list

    def update_primary_reward(self, arm: int, reward: int):
        """ Update first payments alphas and betas.
        Update constraints.

        :param arm:
        :param reward:
        :return:
        """

        self.primary_alphas[arm] += reward
        self.primary_betas[arm] += 1 - reward

        self.n_primary_payments += 1
        self.n_primary_success += reward

        self.constraints[arm] -= reward

    def update_repeated_reward(self, arm: int, reward: int):
        """ Update token payments alphas and betas.

        :param arm:
        :param reward:
        :return:
        """

        self.repeated_alphas[arm] += reward
        self.repeated_betas[arm] += 1 - reward

        self.n_repeated_payments += 1
        self.n_repeated_success += reward

        self.n_payments += 1
        self.n_success += reward

        self.constraints[arm] -= reward

    def update_cascade_reward(self, config: str, reward: int):
        """ Update cascade alphas and betas for particular config.

        :param config:
        :param reward:
        :return:
        """

        self.cascade_alphas[config] += reward
        self.cascade_betas[config] += 1 - reward
        self.update_cascade_mean(config)

        self.n_cascade_payments += 1
        self.n_cascade_success += reward

        self.n_payments += 1
        self.n_success += reward

    def play_cascade(self, arm_list: list):
        """ Cascade routing simulator.

        :param arm_list:
        :return:
        """
        arm_list = [int(x) for x in arm_list.split()]
        # Calculate probability for each step of cascade
        proba_list = self.arms_proba['primary']
        cascade_proba = {arm_list[0]: proba_list[arm_list[0]]}
        cascade_proba.update({arm_list[i + 1]: proba_list[arm_list[i + 1]] - proba_list[arm_list[i]]
                              for i in range(len(arm_list) - 1)})

        check_step = 0
        for k, v in cascade_proba.items():

            reward = self.pull_arm(v)

            if check_step == 0:
                """Update reward for first cascade step"""
                self.update_primary_reward(k, reward)

            if reward == 1:
                """Oneclick payments simulator"""

                repeated_num = int(np.random.gamma(1, 2))
                for i in range(repeated_num):
                    repeated_reward = self.pull_arm(self.arms_proba['repeated'][k])
                    self.update_repeated_reward(k, repeated_reward)

                break

            check_step += 1
        config_string = ' '.join([str(x) for x in arm_list])
        self.update_cascade_reward(config_string, reward)
