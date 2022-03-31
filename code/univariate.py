import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, kstest, linregress, describe, zscore
from sklearn.metrics import roc_curve, roc_auc_score, classification_report

class ContinuousVariable:
    @staticmethod
    def eda(obs, **kwargs):
        temp_data = [i for i in obs if not np.isnan(i)]
        stats = describe(temp_data)
        temp_data = sorted(temp_data, reverse=False)            
        f, axes = plt.subplots(1,3, figsize=(15,5))
        axes[0].hist(temp_data, bins=kwargs.get('hist_bin',15))
        
        min_var = min(temp_data)
        max_var = max(temp_data)
        x_range = np.arange(min_var, max_var, (max_var-min_var)/50)
        xs = [0.5*(x2+x1) for x1,x2 in zip(x_range[:-1], x_range[1:])]
        kde_obs = gaussian_kde(temp_data, bw_method=kwargs.get('kde_smooth_param',1.0))(x_range)
        
        axes[1].plot(x_range, kde_obs)
        
        axes[2].boxplot([temp_data],patch_artist=True)  
        rp = kwargs.get('rounding_precission',4)

        table_data=[
            ["Mean", np.round(stats.mean,rp),"Stddev", np.round(np.sqrt(stats.variance),rp)],
            ["Min", np.round(stats.minmax[0],rp), 
             "IQR",  np.round(temp_data[int(0.75*stats.nobs)] 
                               - temp_data[int(0.25*stats.nobs)],rp)],
            ["1st Percentile", np.round(temp_data[int(0.01*stats.nobs)],rp),
            "Skew", np.round(stats.skewness,rp)],
            ["Q1", np.round(temp_data[int(0.25*stats.nobs)],rp),
            "Kurtosis", np.round(stats.kurtosis,rp)],
            ["Median", np.round(temp_data[int(0.5*stats.nobs)],rp),
            "Missings", len(obs)-len(temp_data)],
            ["Q2", np.round(temp_data[int(0.75*stats.nobs)],rp),
            "Outliers (abs(zscore)>3)", sum([abs(i) > 3 for i in zscore(temp_data)])],
            ["99th Percentile", np.round(temp_data[int(0.99*stats.nobs)],rp),"",""],
            ["Max", np.round(stats.minmax[1],rp),"",""]]
                                                                   
        f, axes = plt.subplots(1,1, figsize=(12,3))
        table = axes.table(cellText=table_data,loc='center')
        axes.axis('off')
        plt.show()
        
    @staticmethod
    def discrimination_analysis(goods, bads, **kwargs):
        hist_all = np.histogram([*goods, *bads], bins=kwargs.get('hist_bin', 15))
        hist_bads = np.histogram(bads, bins=hist_all[1])
        
        f, axes = plt.subplots(3,3, figsize=(25,15))
        # 1st row
        hist_good = axes[0][0].hist(goods, label='good', color='g',alpha=0.5,
                                    bins=hist_all[1],density=True)
        hist_bad = axes[0][0].hist(bads, label='bad', color='r',alpha=0.5,
                                   bins=hist_all[1],density=True)
        axes[0][0].set_title("Histogram")
        axes[0][0].legend()

        min_var = min(hist_good[1][0], hist_bad[1][0])
        max_var = max(hist_good[1][-1], hist_bad[1][-1])
        x_range = np.arange(min_var, max_var, (max_var-min_var)/50)
        xs = [0.5*(x2+x1) for x1,x2 in zip(x_range[:-1], x_range[1:])]

        kde_goods = gaussian_kde(goods, bw_method=kwargs.get('kde_smooth_param',1.0))(x_range)
        kde_bads  = gaussian_kde(bads, bw_method=kwargs.get('kde_smooth_param',1.0))(x_range)
        axes[0][1].plot(x_range, kde_bads, 'r', label='bad')
        axes[0][1].plot(x_range, kde_goods, 'g', label='good')
        axes[0][1].legend()
        axes[0][1].set_title("Probability Distribution Function")

        boxes = axes[0][2].boxplot([goods, bads], labels=['good', 'bad'],patch_artist=True)        
        axes[0][2].set_title("BoxPlots")
        boxes['boxes'][0].set(color='lightgreen', linewidth=2)
        boxes['medians'][0].set(color='k')
        boxes['boxes'][1].set(color='lightcoral', linewidth=2)
        boxes['medians'][1].set(color='k')    
        # 2nd row
        roc_curve_data = roc_curve([0]*len(goods)+[1]*len(bads), 
                                   np.concatenate([goods, bads]))
        axes[1][0].plot(roc_curve_data[0], roc_curve_data[1], 'r-')
        axes[1][0].plot([0.0, 1.0], [0.0, 1.0], 'k--')
        axes[1][0].set_title('ROC Curve')
        
        bads_cdf = [len([b for b in bads if b < x])/len(bads) for x in x_range]
        goods_cdf = [len([b for b in goods if b < x])/len(goods) for x in x_range]
        
        KS = max([((x,i,j),abs(j-i)) for x,i,j in zip(x_range, bads_cdf, goods_cdf)], key=lambda t:t[1])
        axes[1][1].plot(x_range, bads_cdf, 'r-')
        axes[1][1].plot(x_range, goods_cdf, 'g-')
        axes[1][1].plot([KS[0][0], KS[0][0]], [KS[0][1], KS[0][2]], 'k-')
        axes[1][1].set_title('CDF with KS distance')
        
        obs = sorted([*[(i, 0) for i in goods],*[(i, 1) for i in bads]], key=lambda x:x[0], reverse=True)
        auc = roc_auc_score([i[1] for i in obs],[i[0] for i in obs])
        
        table_data=[
            ["AUC", "{:.2%}".format(auc)],
            ["Gini", "{:.2%}".format(2*auc-1)],
            ["Fisher information", "{:.5}".format(
             np.mean(goods)-np.mean(bads)/(np.sqrt((np.std(goods)+np.std(bads))/2)))],
            ["Kolmogorow-Smirnow's D", "{:.2%}".format(kstest(goods, bads)[0])]
        ]   
        table = axes[1][2].table(cellText=table_data,loc='center')
        axes[1][2].axis('off')
        
        # 3rd row          
        drs = [(x ,d/a) for x,d,a in zip(hist_all[1],hist_bads[0], hist_all[0]) if a != 0]
        trendline = np.polyfit([i[0] for i in drs],[i[1] for i in drs], 3)
        poly = np.poly1d(trendline)
        axes[2][0].scatter([i[0] for i in drs],[i[1] for i in drs], marker='x', color='r')
        axes[2][0].plot([i[0] for i in drs],poly([i[0] for i in drs]),"k:")
        axes[2][0].set_title("Default Rate Curve")
        
        logodds = [(x , np.log(b/(a-b))) for x,b,a in zip(hist_all[1],hist_bads[0], hist_all[0]) if  a != b and b != 0]    
        axes[2][1].scatter([i[0] for i in logodds],[i[1] for i in logodds], marker='x', color='r', label='Log Odds')
        axes[2][1].set_title("Log Odds")
        lnreg= linregress([i[0] for i in logodds],[i[1] for i in logodds])
        axes[2][1].plot([i[0] for i in drs],[lnreg.slope*i[0]+lnreg.intercept for i in drs],"k:", 
                        label='Linear R2='+str(round(lnreg.rvalue*100,2))+"%")
        axes[2][1].legend()
        
        axes[2][2].scatter([i[0] for i in logodds],
                           [i[1]-(lnreg.slope*i[0]+lnreg.intercept) for i in logodds], 
                           marker='x', color='r', label='Log Odds')
        axes[2][2].plot([drs[0][0], drs[-1][0]], [0, 0], 'k--')
        axes[2][2].set_title("Log Odds ~ variable Residuals")
        plt.show()    

    @staticmethod
    def compare_discrimination_in_subpops(df, tested_variable, default_flag, subpop_flag):
        d = df[[tested_variable, default_flag, subpop_flag]]\
            .groupby(subpop_flag)\
            .apply(lambda t:roc_curve(t[default_flag], t[tested_variable]))\
            .to_dict()

        fig, axes = plt.subplots(1,1,figsize=(5,4))
        for k,v in d.items():
            axes.plot(v[0],v[1],label=k)
        axes.legend()
        axes.plot([0,1], [0,1], 'k--')
        fig, axes = plt.subplots(1,1,figsize=(10,2))

        auc = df[[tested_variable, default_flag, subpop_flag]]\
            .groupby(subpop_flag)\
            .apply(lambda t:roc_auc_score(t[default_flag], t[tested_variable]))\
            .to_dict()

        table_data=[
            ['Stats'],
            ['AUC'],
            ['Gini']
        ]
        for k,v in auc.items():
            table_data[0].append(k)
            table_data[1].append("{:.2%}".format(v))
            table_data[2].append("{:.2%}".format(2*v-1))

        table = axes.table(cellText=table_data,loc='center')
        axes.axis('off')

        plt.show()
        
        
class DiscreteVariable:
    @staticmethod
    def eda(obs, **kwargs):
        fig, ax = plt.subplots(1,2,figsize=(10,4))
        vals = obs.value_counts().to_dict()
        ax[0].bar(vals.keys(),vals.values())
        ax[0].tick_params(axis='x', labelrotation=90)
        
        ax[1].pie([v / len(obs) for v in vals.values()], labels=vals.keys(), autopct='%1.0f%%')
        

        plt.show()
        
    def discrimination_analysis(goods, bads):
        goods_vals = goods.value_counts().to_dict()
        bads_vals = bads.value_counts().to_dict()
        all_vals = dict([(k,goods_vals[k]+bads_vals[k]) for k in set(goods).union(set(bads))])
        keys_ordered = [k for k,v in sorted(all_vals.items(),
                                            key=lambda x:x[1])]       

        fig, axes = plt.subplots(2,2, figsize=(12,8))
        woe = dict([(k,np.log(bads_vals.get(k,0) / goods_vals.get(k,0))) for k in keys_ordered])
        iv = sum([(np.log(bads_vals.get(k,0) / goods_vals.get(k,0)))\
                  *(bads_vals.get(k,0)-goods_vals.get(k,0))/all_vals[k] 
                  for k in keys_ordered])
        roc_data_vals = [woe[g] for g in goods] + [woe[b] for b in bads]
        auc = roc_auc_score([0]*len(goods)+[1]*len(bads),roc_data_vals)
        roc_curve_data = roc_curve([0]*len(goods)+[1]*len(bads), roc_data_vals)
        tbl_cells = [["AUC",str(round(auc*100,4))+"%"],
                     ["Gini",str(round((2*auc-1)*100,4))+"%"],
                     ["Information Value",round(iv,4)],
                    ]
        axes[0][0].plot(roc_curve_data[0], roc_curve_data[1],'r')
        axes[0][0].plot([0,1], [0,1],'k--')
        axes[0][1].table(cellText=tbl_cells,loc='center')
        axes[0][1].axis('off')

        axes[1][0].bar(keys_ordered, [goods_vals.get(k,0) for k in keys_ordered],color='g')
        axes[1][0].bar(keys_ordered, [bads_vals.get(k,0) for k in keys_ordered],
                      bottom=[goods_vals.get(k,0) for k in keys_ordered],color='r')
        axes[1][0].tick_params(axis='x', labelrotation=90)

        keys_ordered = [i[0] for i in sorted([(k,goods_vals[k]/a) for k,a in all_vals.items()],
                              key=lambda x:x[1])]
        axes[1][1].bar(keys_ordered, [goods_vals.get(k,0) / all_vals[k]
                                      for k in keys_ordered],color='g')
        axes[1][1].bar(keys_ordered, [bads_vals.get(k,0)/all_vals[k] 
                                      for k in keys_ordered],
                      bottom=[goods_vals.get(k,0)/all_vals[k] for k in keys_ordered],color='r')
        axes[1][1].tick_params(axis='x', labelrotation=90)

        plt.show()