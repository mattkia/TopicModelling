import matplotlib.pyplot as plt


perplexities = [57411.685, 88667.155, 107322.932, 130670.090, 149156.922, 167610.489]
num_of_topics = [5, 7, 9, 11, 13, 15]

plt.plot(num_of_topics, perplexities, linestyle='--', marker='o', color='red')
plt.title('Perplexity for Number of Topics')
plt.xlabel('Number of Topics')
plt.ylabel('Perplexity')
plt.show()

