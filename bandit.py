import numpy as np

from environment import TestEnvironment


class Strategy:
    """Base for possible bandit strategies.
    """

    def __init__(self, env: TestEnvironment, cascade_params = ['primary']):
        self.env = env

        self.cascade_params = cascade_params

    def choose_step_arm(self, step_parameter, current_cascade):
        """Particular bandit strategy based on Thompson Sampling.

        Always choose an arm with highest estimate.
        """
        if step_parameter == 'repeated':
            estimation_list = [np.random.beta(self.env.repeated_alphas[i], self.env.repeated_betas[i])
                               if (self.env.constraints[i] > 0) and (i not in current_cascade) else 0
                               for i in range(self.env.n_arms)]
        else:
            estimation_list = [np.random.beta(self.env.primary_alphas[i], self.env.primary_betas[i])
                               if (self.env.constraints[i] > 0) and (i not in current_cascade) else 0
                               for i in range(self.env.n_arms)]
        if len(estimation_list) == 0 or sum(estimation_list) == 0:
            arm = None
        else:
            arm = np.argmax(estimation_list)

        return arm

    def cascade_builder(self):
        """Compound bandit strategy based on Thompson Sampling.

        Always choose an config with highest estimate.
        """
        current_cascade = []
        for step_params in self.cascade_params:
            step_arm = self.choose_step_arm(step_params, current_cascade)

            if step_arm is None:
                break
            current_cascade.append(step_arm)

        return current_cascade

    def choose_cascade(self):

        # Create a new cascade config
        new_cascade_config = self.cascade_builder()
        # Add a new cascade config to cascade config list
        if len(new_cascade_config) != 0:
            config_string = ' '.join([str(x) for x in new_cascade_config])
            self.env.update_cascade_config(config_string)

        # Get TOP5 sorted cascades by mean of the beta distribution
        final_cascade_list = self.env.get_cascade_mean()[:5]

        best_cascade_position = np.argmax([np.random.beta(self.env.cascade_alphas[i], self.env.cascade_betas[i])
                                           for i in final_cascade_list])

        return final_cascade_list[best_cascade_position]

class Bandit:
    """Set up and launch bandit problem solver with current environment and strategy"""

    def __init__(self, strategy: Strategy):
        self.strategy = strategy
        self.env = self.strategy.env

    def action(self):
        cascade = self.strategy.choose_cascade()
        self.env.play_cascade(cascade)