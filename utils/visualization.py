import matplotlib.pyplot as plt
import numpy as np
from graphviz import Digraph
plt.ion() 
class Plotter:
    def __init__(self, data, result, sp):
        self.data = data
        self.result = result
        self.sp = sp
    
    def plot_groupcount(self, title):
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'olive', 'cyan', 'magenta']  
        for key, value in self.result.g_count.items():
            ax.plot(value, label=f'Group {key+1}',color=colors[key % len(colors)])      
        ax.set_title(title)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Group sample size')
        ax.set_xlim(0, self.result.iteration)
        ax.set_ylim(0, max(max(v) for v in self.result.g_count.values())+200) 
        ax.legend()  

        plt.show()

    def compare_groups_location(self):
        fig, ax = plt.subplots(1,2,figsize=(10, 4))

        ax[0].scatter(self.sp[:,0], self.sp[:,1],c=self.result.initial_groups) 
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_title(f'Initial Group Results') 

        ax[1].scatter(self.sp[:,0], self.sp[:,1],c=self.result.groups_indexs) 
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_title(f'Final Group Results') 
        ax[1].legend() 

        plt.tight_layout() 
        plt.show()
        
    def each_group_location(self,title='',save_fig=False):
        group_size = max(self.result.groups_indexs)+1
        width_per_subplot = 4  
        height = 4  
        fig_width = width_per_subplot * group_size
        fig, axs = plt.subplots(1, group_size, figsize=(fig_width, height))
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'olive', 'cyan', 'magenta']
        for key in range(group_size):
            ax = axs[key] 
            indices = np.where(self.result.groups_indexs == key)[0]
            # group=self.data.values[indices]
            ax.scatter(x=self.sp[:,0],y=self.sp[:,1],label=f'Group {key+1}',color='gray',alpha=0.2, s= 10)  # plot all data points
            ax.scatter(x=self.sp[indices,0],y=self.sp[indices,1],color=colors[key % len(colors)], label=f'Group {key+1}',s= 10,alpha=0.6, linewidth=0.3,edgecolors='white') # plot group data points               
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            # ax.set_xticks([])
            # ax.set_yticks([])
            ax.set_title(f'Group {key+1}')  
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout() 
        if save_fig:
            plt.savefig('../'+title+'.png', dpi=300)
        else:
            plt.show()


    def plot_diagram(self, group_number, save_fig=False, filename='diagram', cut_off=0.3, label=True):
        '''
        Plot the diagram of the factor loadings for a specific group.
        Parameters:
        - group_number: int, the group number to plot the diagram for.
        - save_fig: bool, whether to save the figure as a png file.
        - filename: str, the name of the file to save the figure as.
        - cut_off: float, the cut-off value for the factor loadings.
        - label: bool, whether to label the edges with the factor loadings.
        '''
        loadings = self.result.Load_Matrix_dict[group_number]

        # Create a directed graph
        dot = Digraph()
        
        factors = [f'Factor {u+1}' for u in range(loadings.shape[1])]
        # Add factors to the graph
        for u in range(loadings.shape[1]):
            dot.node(f'Factor {u+1}', f'Factor {u+1}', shape='ellipse', fillcolor='lightgray', style='filled')

        # Add variables to the graph
        for i, row in enumerate(loadings):
            variable = f'Variable {i+1}'
            for j, loading in enumerate(row):
                if abs(loading) > cut_off:  # Only add edges with absolute loading values greater than cut_off
                    dot.node(variable, variable, shape='box',fixedsize='True',fontsize='10')
                    edge_style = 'dashed' if loading < 0 else 'solid'
                    edge_color = 'red' if loading < 0 else 'black'
                    if label:
                        dot.edge(factors[j], variable, label=f'{loading:.2f}', style=edge_style,fontcolor=edge_color,fontsize='10',decorate='True',lp='True',color=edge_color, minlen='3', headport='n', tailport='c')
                    else:
                        dot.edge(factors[j], variable, style=edge_style, color=edge_color, minlen='3', headport='n', tailport='c')
        
        if save_fig:
            dot.attr(ordering='in', nodesep='0', ranksep='0.5', dpi='300')
            dot.render(filename, format='png', cleanup=True)
        else:
            dot.attr(ordering='in', nodesep='0', ranksep='0.5')
            return dot
