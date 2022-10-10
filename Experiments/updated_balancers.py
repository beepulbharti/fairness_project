import pandas as pd
import numpy as np
import scipy as sp
import itertools
import seaborn as sns

from matplotlib import pyplot as plt
from itertools import combinations
from copy import deepcopy

import updated_tools

class BinaryBalancer:
    def __init__(self,
                 y,
                 y_,
                 a,):
            
        # Setting the variables
        self.y = y
        self.y_ = y_
        self.a = a

        # Getting the group for A = 0 and A = 1
        self.groups = np.unique(a)
        group_ids = [np.where(a == g)[0] for g in self.groups]

        # P(A=0) and P(A=1)
        self.p = [len(cols) / len(y) for cols in group_ids]

        # Calcuating the groupwise classification rates for A = 0 and A = 1
        self.gr_list = [updated_tools.CLFRates(self.y[i], self.y_[i]) 
                         for i in group_ids]
        self.group_rates = dict(zip(self.groups, self.gr_list))
        
        # And then the overall rates
        self.overall_rates = updated_tools.CLFRates(self.y, self.y_)
    
    def adjust(self,
               goal='odds',
               round=4,
               imbalanced = True,
               return_optima=False,
               summary=True,
               binom=False):
        
        self.goal = goal

        # Setting loss 
        if imbalanced == True:
            l_10 = 0.5*(1/(self.overall_rates.num_neg))
            l_01 = 0.5*(1/(self.overall_rates.num_pos))
        else:
            l_10 = 1
            l_01 = 1

        # Getting the coefficients for the linear program
        coefs = [((l_10 * g.tnr * g.num_neg - l_01 * g.fnr * g.num_pos)*self.p[i],
               (l_10 * g.fpr * g.num_neg - l_01 * g.tpr * g.num_pos)*self.p[i])
               for i, g in enumerate(self.gr_list)]
        
        # Setting up the coefficients for the objective function
        obj_coefs = np.array(coefs).flatten()
        obj_bounds = [(0, 1)]
        
        # Constraint matrix Ax = 0
        g0 = self.gr_list[0]
        g1 = self.gr_list[1]
        A = np.zeros((2,4))
        A[0,0] = 1 - g0.fpr
        A[0,1] = g0.fpr
        A[0,2] = -(1-g1.fpr)
        A[0,3] = -g1.fpr
        A[1,0] = 1 - g0.tpr
        A[1,1] = g0.tpr
        A[1,2] = -(1-g1.tpr)
        A[1,3] = -g1.tpr
        self.con = A
        con_b = np.zeros(self.con.shape[0])
        
        # Running the optimization
        self.opt = sp.optimize.linprog(c=obj_coefs,
                                       bounds=obj_bounds,
                                       A_eq=self.con,
                                       b_eq=con_b,
                                       method='highs')
        self.pya = self.opt.x.reshape(len(self.groups), 2)
        
        # Setting the adjusted predictions
        self.y_adj = updated_tools.pred_from_pya(y_=self.y_, 
                                         a=self.a,
                                         pya=self.pya, 
                                         binom=binom)
        
        # Getting theoretical (no rounding) and actual (with rounding) loss
        self.actual_loss = 1 - updated_tools.CLFRates(self.y, self.y_adj).acc
        cmin = self.opt.fun
        
        # Calculating the theoretical balance point in ROC space
        p0, p1 = self.pya[0][0], self.pya[0][1]
        group = self.group_rates[self.groups[0]]
        fpr = (group.tnr * p0) + (group.fpr * p1)
        tpr = (group.fnr * p0) + (group.tpr * p1)
        self.roc = (np.round(fpr, round), np.round(tpr, round))
        
        if summary:
            self.summary(org=False)
        
        if return_optima:                
            return {'loss': self.theoretical_loss, 'roc': self.roc}
        
        
    def predict(self, y_, a, binom=False):
        """Generates bias-adjusted predictions on new data.
        
        Parameters
        ----------
        y_ : ndarry of shape (n_samples,)
            A binary- or real-valued array of unadjusted predictions.
        
        a : ndarray of shape (n_samples,)
            The protected attributes for the samples in y_.
        
        binom : bool, default False
            Whether to generate adjusted predictions by sampling from a \
            binomial distribution.
        
        Returns
        -------
        y~ : ndarray of shape (n_samples,)
            The adjusted binary predictions.
        """
        # Optional thresholding for continuous predictors
        if np.any([0 < x < 1 for x in y_]):
            group_ids = [np.where(a == g)[0] for g in self.groups]
            y_ = deepcopy(y_)
            for g, cut in enumerate(self.cuts):
                y_[group_ids[g]] = updated_tools.threshold(y_[group_ids[g]], cut)
        
        # Returning the adjusted predictions
        adj = updated_tools.pred_from_pya(y_, a, self.pya, binom)
        return adj
    
    def plot(self, 
             s1=50,
             s2=50,
             preds=False,
             optimum=True,
             lp_lines='all', 
             palette='colorblind',
             style='white',
             xlim=(0, 1),
             ylim=(0, 1)):
            
        """Generates a variety of plots for the PredictionBalancer.
        
        Parameters
        ----------
        s1, s2 : int, default 50
            The size parameters for the unadjusted (1) and adjusted (2) ROC \
            coordinates.
        
        preds : bool, default False
            Whether to observed ROC values for the adjusted predictions (as \
            opposed to the theoretical optima).
        
        optimum : bool, default True
            Whether to plot the theoretical optima for the predictions.
        
        roc_curves : bool, default True
            Whether to plot ROC curves for the unadjusted scores, when avail.
        
        lp_lines : {'upper', 'all'}, default 'all'
            Whether to plot the convex hulls solved by the linear program.
        
        shade_hull : bool, default True
            Whether to fill the convex hulls when the LP lines are shown.
        
        chance_line : bool, default True
            Whether to plot the line ((0, 0), (1, 1))
        
        palette : str, default 'colorblind'
            Color palette to pass to Seaborn.
        
        style : str, default 'dark'
            Style argument passed to sns.set_style()
        
        alpha : float, default 0.5
            Alpha parameter for scatterplots.
        
        Returns
        -------
        A plot showing shapes were specified by the arguments.
        """
        # Setting basic plot parameters
        plt.xlim(xlim)
        plt.ylim(ylim)
        sns.set_theme()
        sns.set_style(style)
        cmap = sns.color_palette(palette, as_cmap=True)
        
        # Plotting the unadjusted ROC coordinates
        orig_coords = updated_tools.group_roc_coords(self.y, 
                                             self.y_, 
                                             self.a)
        sns.scatterplot(x=orig_coords.fpr,
                        y=orig_coords.tpr,
                        hue=self.groups,
                        s=s1,
                        palette='colorblind')
        plt.legend(loc='lower right')
        
        # Plotting the adjusted coordinates
        if preds:
            adj_coords = updated_tools.group_roc_coords(self.y, 
                                                self.y_adj, 
                                                self.a)
            sns.scatterplot(x=adj_coords.fpr, 
                            y=adj_coords.tpr,
                            hue=self.groups,
                            palette='colorblind',
                            marker='x',
                            legend=False,
                            s=s2,
                            alpha=1)
        
        # Adding lines to show the LP geometry
        if lp_lines:
            # Getting the groupwise coordinates
            group_rates = self.group_rates.values()
            group_var = np.array([[g]*3 for g in self.groups]).flatten()
            
            # Getting coordinates for the upper portions of the hulls
            upper_x = np.array([[0, g.fpr, 1] for g in group_rates]).flatten()
            upper_y = np.array([[0, g.tpr, 1] for g in group_rates]).flatten()
            upper_df = pd.DataFrame((upper_x, upper_y, group_var)).T
            upper_df.columns = ['x', 'y', 'group']
            upper_df = upper_df.astype({'x': 'float',
                                        'y': 'float',
                                        'group': 'str'})
            # Plotting the line
            sns.lineplot(x='x', 
                         y='y', 
                         hue='group', 
                         data=upper_df,
                         alpha=0.75, 
                         legend=False)
            
            # Optionally adding lower lines to complete the hulls
            if lp_lines == 'all':
                lower_x = np.array([[0, 1 - g.fpr, 1] 
                                    for g in group_rates]).flatten()
                lower_y = np.array([[0, 1 - g.tpr, 1] 
                                    for g in group_rates]).flatten()
                lower_df = pd.DataFrame((lower_x, lower_y, group_var)).T
                lower_df.columns = ['x', 'y', 'group']
                lower_df = lower_df.astype({'x': 'float',
                                            'y': 'float',
                                            'group': 'str'})
                # Plotting the line
                sns.lineplot(x='x', 
                             y='y', 
                             hue='group', 
                             data=lower_df,
                             alpha=0.75, 
                             legend=False)       
        
        # Optionally adding the post-adjustment optimum
        if optimum:
            if self.roc is None:
                print('.adjust() must be called before optimum can be shown.')
                pass
            
            elif 'odds' in self.goal:
                plt.scatter(self.roc[0],
                                self.roc[1],
                                marker='x',
                                color='black')
        
        plt.show()
    
    def summary(self, org=True, adj=True):
        """Prints a summary with FPRs and TPRs for each group.
        
        Parameters:
            org : bool, default True
                Whether to print results for the original predictions.
            
            adj : bool, default True
                Whether to print results for the adjusted predictions.
        """
        if org:
            org_coords = updated_tools.group_roc_coords(self.y, self.y_, self.a)
            org_loss = 1 - self.overall_rates.acc
            print('\nPre-adjustment group rates are \n')
            print(org_coords.to_string(index=False))
            print('\nAnd loss is %.4f\n' %org_loss)
        
        if adj:
            adj_coords = updated_tools.group_roc_coords(self.y, self.y_adj, self.a)
            adj_loss = 1 - updated_tools.CLFRates(self.y, self.y_adj).acc
            print('\nPost-adjustment group rates are \n')
            print(adj_coords.to_string(index=False))
            print('\nAnd loss is %.4f\n' %adj_loss)
