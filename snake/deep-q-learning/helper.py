import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores, mean_scores, losses=None):
    display.clear_output(wait=True)
    #plt.show()
    plt.clf()
    plt.title('Training Progress')
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')
    
    if losses is not None and len(losses) > 0:
        plt.plot(losses, label='Loss', linestyle='dotted')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.pause(0.1)

def plot_reward(plot_total_rewards, plot_mean_rewards):
    plt.plot(plot_total_rewards, label='Reward')
    plt.plot(plot_mean_rewards, label='Mean Reward')
    plt.legend()
    plt.title("Reward per Game")
    plt.xlabel("Game")
    plt.ylabel("Reward")
    plt.savefig("reward_plot.png")
    plt.close()