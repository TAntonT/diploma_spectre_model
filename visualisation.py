"""
Draw bandit behaviour at environment.
"""

import numpy as np
import scipy.stats as stats
from matplotlib.cm import get_cmap

from operator import itemgetter

from bandit import Bandit


class DrawEnvironment():

    def __init__(self, ax, bandit: Bandit):

        self.bandit = bandit

        if self.bandit.env.failure is True:
            self.failure_message_text = ''

        self.prob = bandit.env.arms_proba
        self.n_arms = bandit.env.n_arms

        self.x = np.linspace(0, 1, 500)
        self.y_max = 5

        self.cm = get_cmap('tab10')

        """Draw primary payments plot"""
        self.ax1 = ax[0, 0]
        self.line1 = [self.ax1.plot([], [])[0] for _ in range(self.n_arms)]

        # Set up plot parameters
        self.ax1.set_xlim(0, 1)
        self.ax1.set_ylim(0, self.y_max)
        self.ax1.get_yaxis().set_visible(False)
        self.ax1.grid(True)
        self.ax1.autoscale(enable=True)
        self.ax1.title.set_text('Primary payments')

        # Set up text
        self.iter_text1 = self.ax1.text(0.02, 0.02, '', transform=self.ax1.transAxes)

        # Set up a color palette
        self.ax1.set_prop_cycle(color=get_cmap('tab10').colors)

        # This vertical line represents the theoretical value, to
        # which the plotted distribution should converge.
        for arm in range(self.n_arms):
            self.ax1.axvline(self.prob['primary'][arm], linestyle='--', color=self.cm(arm), label='bandit_' + str(arm))

        self.ax1.legend(loc='upper left')

        """Draw repeated payments plot"""
        self.ax2 = ax[0, 1]
        self.line2 = [self.ax2.plot([], [])[0] for _ in range(self.n_arms)]

        # Set up plot parameters
        self.ax2.set_xlim(0, 1)
        self.ax2.set_ylim(0, self.y_max)
        self.ax2.get_yaxis().set_visible(False)
        self.ax2.grid(True)
        self.ax2.autoscale(enable=True)
        self.ax2.title.set_text('Repeated payments')

        # Set up text
        self.iter_text2 = self.ax2.text(0.02, 0.02, '', transform=self.ax2.transAxes)

        # Set up a color palette
        self.ax2.set_prop_cycle(color=get_cmap('tab10').colors)

        # This vertical line represents the theoretical value, to
        # which the plotted distribution should converge.
        for arm in range(self.n_arms):
            self.ax2.axvline(self.prob['repeated'][arm], linestyle='--', color=self.cm(arm), label='bandit_' + str(arm))

        self.ax2.legend(loc='upper left')

        """General stats"""
        self.ax3 = ax[1, 0]

        # Set up plot parameters
        self.ax3.title.set_text('General statistics')
        self.ax3.axis('off')

        # Set up text
        self.iter_text3 = self.ax3.text(0.02, 0.02, '', transform=self.ax3.transAxes)

        # Set up a color palette
        self.ax3.set_prop_cycle(color=get_cmap('tab10').colors)

        """Cascade full list"""
        self.ax4 = ax[1, 1]

        self.table_text = [[None, None, None, None, None]]

        # Set up plot parameters
        self.ax4.title.set_text('Cascade config table')
        self.ax4.axis('off')
        self.table4 = self.ax4.table(cellText = self.table_text,
            colLabels=['Cascade\nConfig',
                       'Estimated\nConversion\n(Mean)',
                       'Alpha',
                       'Beta',
                       'Payment\nNumber'],
            bbox=[0, 0, 1, 1])

    def set_dist_params(self):

        self.primary_alphas = self.bandit.env.primary_alphas
        self.primary_betas = self.bandit.env.primary_betas

        self.repeated_alphas = self.bandit.env.repeated_alphas
        self.repeated_betas = self.bandit.env.repeated_betas

        self.cascade_alphas = self.bandit.env.cascade_alphas
        self.cascade_betas = self.bandit.env.cascade_betas

        return self.primary_alphas, self.primary_betas, self.repeated_alphas, \
               self.repeated_betas, self.cascade_alphas, self.cascade_betas

    def __call__(self, i):
        print(i)
        alpha_beta = self.set_dist_params()

        if self.bandit.env.failure is True:
            # Bank failure simulation
            failure_message = self.bandit.env.delete_bank()
            if failure_message is None:
                failure_message = self.bandit.env.add_bank()

        if self.bandit.env.n_primary_payments > 0:
            conv = round(self.bandit.env.n_primary_success / float(self.bandit.env.n_primary_payments) * 100, 2)
        else:
            conv = 0

        text1 = 'payments = ' + str(self.bandit.env.n_primary_payments) + \
                '\nsuccess = ' + str(self.bandit.env.n_primary_success) + \
                '\nconversion = ' + str(conv) + '%'

        if self.bandit.env.n_repeated_payments > 0:
            conv = round(self.bandit.env.n_repeated_success / float(self.bandit.env.n_repeated_payments) * 100, 2)
        else:
            conv = 0

        text2 = 'payments = ' + str(self.bandit.env.n_repeated_payments) + \
                '\nsuccess = ' + str(self.bandit.env.n_repeated_success) + \
                '\nconversion = ' + str(conv) + '%'

        if self.bandit.env.n_payments > 0:
            conv = round(self.bandit.env.n_success / float(self.bandit.env.n_payments) * 100, 2)
        else:
            conv = 0

        if self.bandit.env.n_cascade_payments > 0:
            conv_cascade = round(self.bandit.env.n_cascade_success / float(self.bandit.env.n_cascade_payments) * 100, 2)
        else:
            conv_cascade = 0

        text3 = 'iter = ' + str(i) + \
                '\npayments = ' + str(self.bandit.env.n_payments) + \
                '\nsuccess = ' + str(self.bandit.env.n_success) + \
                '\nconversion = ' + str(conv) + '%' + \
                '\n\ncascade_payments = ' + str(self.bandit.env.n_cascade_payments) + \
                '\ncascade_success = ' + str(self.bandit.env.n_cascade_success) + \
                '\ncascade_conversion = ' + str(conv_cascade) + '%' + \
                '\nconstraints:'
        for i in range(self.n_arms):
            text3 += '\n bandit_' + str(i) + '= ' + str(self.bandit.env.constraints[i])

        text3 += '\n\ncascade_parameters: ' + str(self.bandit.strategy.cascade_params)
        text3 += '\ncascade_configs: ' + str(self.bandit.env.get_cascade_mean()[:5])

        if self.bandit.env.failure is True:
            if failure_message is not None:
                self.failure_message_text = failure_message
            text3 += '\n' + self.failure_message_text

        self.table_text = [[c,
                           str(round(self.bandit.env.cascade_mean[c] * 100, 1)) + '%',
                           self.bandit.env.cascade_alphas[c],
                           self.bandit.env.cascade_betas[c],
                           self.bandit.env.cascade_alphas[c] + self.bandit.env.cascade_betas[c]]
                           for c in self.bandit.env.historical_cascade_config]

        if len(self.table_text) > 0:
            self.table4 = self.ax4.table(cellText=sorted(self.table_text,key=lambda x: x[4], reverse = True)[:9],
                                         colLabels=['Cascade\nConfig',
                                                    'Estimated\nConversion',
                                                    'Alpha',
                                                    'Beta',
                                                    'Payment\nNumber'],
                                         bbox=[0, 0, 1, 1])

        self.iter_text1.set_text(text1)
        self.iter_text2.set_text(text2)
        self.iter_text3.set_text(text3)

        for lnum1, line1 in enumerate(self.line1):
            y = stats.beta.pdf(self.x, alpha_beta[0][lnum1], alpha_beta[1][lnum1], 0, 1)
            if y.max() > self.y_max:
                self.y_max = y.max() + 5
            line1.set_data(self.x, y)

        for lnum2, line2 in enumerate(self.line2):
            y = stats.beta.pdf(self.x, alpha_beta[2][lnum2], alpha_beta[3][lnum2], 0, 1)
            if y.max() > self.y_max:
                self.y_max = y.max() + 5
            line2.set_data(self.x, y)

        self.ax1.set_ylim(0, self.y_max)
        self.ax1.set_xlim(0, 1)

        self.ax2.set_ylim(0, self.y_max)
        self.ax2.set_xlim(0, 1)

        self.bandit.action()

        return tuple(self.line1) + tuple(self.line2) + tuple([self.table4]) + \
               (self.iter_text1, self.iter_text2, self.iter_text3)
