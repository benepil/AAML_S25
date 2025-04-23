import matplotlib.pyplot as plt
from IPython import display

plt.ion()

scores = []
mean_scores = []

def plot(scores_list, mean_scores_list):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores_list, label='Score')
    plt.plot(mean_scores_list, label='Mean Score')
    plt.ylim(ymin=0)
    plt.legend()
    plt.text(len(scores_list)-1, scores_list[-1], str(scores_list[-1]))
    plt.text(len(mean_scores_list)-1, mean_scores_list[-1], str(mean_scores_list[-1]))
    plt.pause(0.1)
