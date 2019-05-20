import matplotlib.pyplot as plt
import pandas as pd
from math import pi
 
# Set data

 
 
def SetupPlot(counters, filenames):
    # ------- PART 1: Create background

    df = pd.DataFrame({
    'Method': ['kNN','Forest'],
    'Норма': [counters[0][0], counters[1][0]],
    'I-ая стадия': [counters[0][1], counters[1][1]],
    'II-ая стадия': [0, 0],
    'III-ая стадия': [0, 0],
    'IV-ая стадия': [0, 0]
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
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([1,2,5], ["1","2","5"], color="grey", size=7)
    plt.ylim(0,6)
    
    
    # ------- PART 2: Add plots
    
    # Plot each individual = each line of the data
    # I don't do a loop, because plotting more than 3 methods makes the chart unreadable
    
    # Ind1
    values=df.loc[0].drop('Method').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="kNN")
    ax.fill(angles, values, 'b', alpha=0.1)
    
    # Ind2
    values=df.loc[1].drop('Method').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="Random Forest")
    ax.fill(angles, values, 'r', alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.show()

if __name__ == "__main__":
    SetupPlot(0,0)