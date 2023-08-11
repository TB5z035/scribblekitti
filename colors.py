import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


color = [[0, 0, 0], [174, 199, 232]]
color = np.array(color)/255.
labels = ["unlabeled", "car"]
patches = [ mpatches.Patch(color=color[i], label="{:s}".format(labels[i]) ) for i in range(len(color)) ] 

ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height*0.8])

plt.legend(handles=patches, fontsize='x-small', bbox_to_anchor=(0.95,1.12), ncol=20)

plt.savefig("color.jpg")