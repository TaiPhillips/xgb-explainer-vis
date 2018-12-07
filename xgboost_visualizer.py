# import
import numpy as np
import pandas as pd

import xgboost as xgb

import matplotlib.pyplot as plt


# ===================================================================================================================
# functions to get logit contributions
# taken from: https://github.com/gameofdimension/xgboost_explainer
# ===================================================================================================================
def check_params(tree, eta, lmda):
    right = tree[-1]
    left = tree[-2]
    assert left['is_leaf'] == True
    assert right['is_leaf'] == True
    assert left['parent'] == right['parent']
    parent = tree[left['parent']]

    Hl = left['cover']
    Hr = right['cover']
    Gl = -1.*left['leaf']*(Hl+lmda)/eta
    Gr = -1.*right['leaf']*(Hr+lmda)/eta

    Gp = Gl + Gr
    Hp = Hl + Hr
    expect_gain = Gl**2/(Hl+lmda) + Gr**2/(Hr+lmda) - Gp**2/(Hp+lmda)
    # print(expect_gain, parent['gain'])
    assert abs(expect_gain-parent['gain']) < 1.e-2

def model2table(bst, eta=0.3, lmda=1.0):
    lst_str = bst.get_dump(with_stats=True)
    tree_lst = [[] for _ in lst_str]
    for i,line in enumerate(lst_str):
        # print(i, line)
        tree_idx = i
        parent = {}
        parent[0] = None
        lst_node_str = line.split('\n')
        node_lst = [{} for _ in range(len(lst_node_str)-1)]
        for node in lst_node_str:
            node = node.strip()
            # print("fdfdf",len(node))
            if len(node) <= 0:
                continue
            is_leaf=False
            if ":leaf=" in node:
                is_leaf=True
            # print(segs[0], segs[1])
            node_idx = int(node[:node.index(":")])
            # print(node_idx)
            d = {}
            d['tree'] = tree_idx
            d['node'] = node_idx
            d['is_leaf'] = is_leaf
            if not is_leaf:
                segs = node.split(' ')
                fl = node.index('[')
                fr = node.index('<')
                d['feature'] = node[fl+1:fr]
                for p in segs[1].split(','):
                    k,v = p.split('=')
                    d[k]=v
                d['yes'] = int(d['yes'])
                d['no'] = int(d['no'])
                d['missing'] = int(d['missing'])
                parent[d['yes']] = node_idx
                parent[d['no']] = node_idx
                d['gain'] = float(d['gain'])
                d['cover'] = float(d['cover'])
            else:
                _, lc = node.split(':')
                for p in lc.split(','):
                    k,v = p.split('=')
                    d[k]=v
                d['leaf'] = float(d['leaf'])
                d['cover'] = float(d['cover'])

            # node_lst.append(d)
            node_lst[node_idx] = d
        for j, node in enumerate(node_lst):
            node_lst[j]['parent'] = parent[node_lst[j]['node']]
        tree_lst[i] = node_lst
    for t in tree_lst:
        check_params(t, eta, lmda)
        for j in reversed(range(len(t))):
            node = t[j]
            if node['is_leaf']:
                G = -1.*node['leaf']*(node['cover']+lmda)/eta
            else:
                G = t[node['yes']]['grad'] + t[node['no']]['grad']
            t[j]['grad'] = G
            t[j]['logit'] = -1.*G/(node['cover']+lmda)*eta
    for t in tree_lst:
        for j in reversed(range(len(t))):
            node = t[j]
            if node['parent'] is None:
                node['logit_delta'] = node['logit'] - .0
            else:
                node['logit_delta'] = node['logit'] - t[node['parent']]['logit']

    return tree_lst

def logit_contribution(tree_lst, leaf_lst):
    dist = {'intercept':0.0}
    for i, leaf in enumerate(leaf_lst):
        tree = tree_lst[i]
        node = tree[leaf]
        parent_idx = node['parent']
        # print(node, parent_idx)
        while True:
            if parent_idx is None:
               dist['intercept'] += node['logit_delta']
               break
            else:
                parent = tree[parent_idx]
                feat = parent['feature']
                if not feat in dist:
                    dist[feat] = 0.0
                dist[feat] += node['logit_delta']
                node = tree[parent_idx]
                parent_idx = node['parent']
    return dist


# ===================================================================================================================
# plotting function
# ===================================================================================================================

def logistic(x):
    return 1 / (1 + np.exp(-x))


def plot_contribution(model, sample, features):
    '''Takes the trained xgboost model using xgboost.train() and a sample which is a xgb.DMatrix.
    Produce a plot explaining the final probability by feature breakdowns'''

    # prepare inference tree
    tree_lst = model2table(model)

    # predict on sample and get contribution
    sample_pred = model.predict(sample, pred_leaf=True)
    dist = logit_contribution(tree_lst, sample_pred[0])
    # print(dist)

    # obtain logit contributions
    sum_logit = 0.0 # <- np.exp(sum_logit) will be the final prediction
    feature_order = []
    logit_contrib_order = []
    for k in dist:
        sum_logit += dist[k]
        fn = features[int(k[1:])] if k != "intercept" else k
        feature_order.append(fn)
        logit_contrib_order.append(dist[k])
        # print(fn + ":", dist[k])

    # organize data and sort by absolute contribution in descending order
    contrib_df = pd.DataFrame({"feature": feature_order,
                               "contrib": logit_contrib_order})
    contrib_df = contrib_df.reindex(contrib_df.contrib.abs().sort_values(ascending=False).index)

    # get numbers in easier accessible variables
    intercept_contrib = contrib_df["contrib"][contrib_df["feature"] == "intercept"].iloc[0]
    feats = np.array(["intercept"] + list(contrib_df["feature"][contrib_df["feature"] != "intercept"]) + ["final"])
    contribs = np.array([intercept_contrib] + list(contrib_df["contrib"][contrib_df["feature"] != "intercept"]))

    # intercept bar
    # print(logistic(contribs[0])-0.5)
    plt.bar(0, logistic(contribs[0])-0.5, bottom=0.5, width=0.9, color="green" if contribs[0] > 0 else "red", edgecolor='black')

    # all variable bars
    for i in range(1, len(contribs)):
        this_bottom = logistic(contribs[:i].sum())
        next_bottom = logistic(contribs[:i+1].sum())
        # print(next_bottom-this_bottom)
        plt.bar(i, next_bottom-this_bottom, bottom=this_bottom, width=0.9,
                color="green" if contribs[i] > 0 else "red", edgecolor='black')

    # last bar (the final logit value)
    print("Final logit contribution: {}, predicted probability: {}".format(sum_logit, logistic(sum_logit)))
    plt.bar(len(contribs), logistic(contrib_df["contrib"].sum())-0.5, bottom=0.5, width=0.9, color="black", edgecolor='black')

    # add a horizontal dot line at 0.5
    plt.plot([0,len(contribs)],[0.5,0.5], 'k--', lw=1)

    plt.ylim([-0.2, 1.2])
    plt.xticks(range(len(contribs)+1), feats, rotation=45, ha="right")
    plt.xlabel("Features ordered by absolute contribution")
    plt.ylabel("Probability")
    plt.title("Breakdown of XGBoost Prediction by Feature-wise Contribution")
