import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import numpy as np
# Set data

 
 
def SetupPlot(counters, filenames):
    # ------- PART 1: Create background

    df = pd.DataFrame({
    'Method': ['kNN','Forest'],
    'Норма': [counters[0][0], counters[1][0]],
    'I-ая стадия': [counters[0][1], counters[1][1]],
    'II-ая стадия': [counters[0][2], counters[1][2]],
    'III-ая стадия': [counters[0][3], counters[1][3]],
    'IV-ая стадия': [counters[0][4], counters[1][4]]
    })
    # number of variable
    categories=list(df)[1:]
    N = len(categories)
    
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)
    
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories)
    max = np.amax(counters)
    if max < 5:
        max = 6
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([1,np.amin(counters),int(np.amax(counters)/2),np.amax(counters)+1], ["1",str(np.amin(counters)),str(int(np.amax(counters)/2)),str(np.amax(counters)+1)], color="grey", size=7)
    plt.ylim(0,6)
    
    
    # ------- PART 2: Add plots
    
    # Plot each individual = each line of the data
    # I don't do a loop, because plotting more than 3 methods makes the chart unreadable
    
    # Ind1
    #values=df.loc[0].drop('Method').values.flatten().tolist()
    #values += values[:1]
    #ax.plot(angles, values, linewidth=1, linestyle='solid', label="kNN")
    #ax.fill(angles, values, 'b', alpha=0.1)
    
    # Ind2
    values=df.loc[1].drop('Method').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="Количество снимков")
    ax.fill(angles, values, 'r', alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.show()

if __name__ == "__main__":
    SetupPlot(0,0)