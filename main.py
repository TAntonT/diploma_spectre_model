import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegFileWriter

import bandit
from visualisation import DrawEnvironment

def main():
    arms_primary_prob = [0.2, 0.72, 0.83, 0.7, 0.75]
    # setup payment by token probability of each arm in environment
    arms_repeated_prob = [0.8, 0.7, 0.7, 0.4, 0.71]
    # setup capacity of success payments for each arm
    constraints = [1000, 100, 75, 120, 50]
    # setup number of iterations (payments) in simulation
    n_max_iters = 400
    # cascade config
    #cascade_params = ['primary']
    #cascade_params = ['repeated']
    #cascade_params = ['primary', 'primary']
    cascade_params = ['repeated', 'primary']

    # setup environment
    testenv = bandit.TestEnvironment(arms_primary_prob, arms_repeated_prob, constraints, failure=False)

    # setup strategy
    strategy = bandit.Strategy(testenv, cascade_params = cascade_params)

    cascade_bandit = bandit.Bandit(strategy)

    fig, ax = plt.subplots(2, 2, figsize = (20, 12))

    draw_env = DrawEnvironment(ax, cascade_bandit)

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = FuncAnimation(fig,
                         draw_env,
                         frames=range(n_max_iters),
                         interval=100,
                         repeat=False,
                         blit=True)

    plt.rcParams['animation.ffmpeg_path'] = '/Users/anton/Documents/ffmpeg'
    mywriter = FFMpegFileWriter(fps=20)
    anim.save("test.mpeg", writer=mywriter)

    #plt.show()



if __name__ == '__main__':
    main()
