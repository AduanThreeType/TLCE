
import pandas as pd
import numpy as np
from graph import NeighborFinder




def load_real_data(dataName):
    g_df = pd.read_csv('./data/test/Real/processed/ml_{}.csv'.format(dataName))
    src_list = g_df.u.values
    dst_list = g_df.i.values
    ts_list = g_df.ts.values

    max_idx = max(g_df.u.values.max(), g_df.i.values.max())
    node_count = len(set(np.unique(np.hstack([g_df.u.values, g_df.i.values]))))
    node_list = np.unique(np.hstack([src_list, dst_list]))
    maxTime_list = max(g_df.ts.values)
    adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(src_list, dst_list, g_df.idx.values, ts_list):
        adj_list[dst].append((src, eidx, ts))
    #每个数据集有个对应的邻居搜索器 其输入是一个类似于邻接矩阵的处理，每个节点行对应其事件三元组 入节点的id 事件id 以及事件的时间戳
    ngh_finder = NeighborFinder(adj_list, uniform=False)
    return src_list, dst_list, ts_list, node_count, node_list, maxTime_list, ngh_finder


def load_train_real_data(UNIFORM):
    src_list, dst_list, ts_list, node_count, node_list, maxTime_list, ngh_finder = [], [], [], [], [], [], []
    train_real_datasets = ['edit-mrwiktionary', 'edit-siwiktionary', 'edit-stwiktionary', 'edit-wowiktionary',
                           'edit-tkwiktionary', 'edit-aywiktionary', 'edit-anwiktionary', 'edit-pawiktionary',
                           'edit-iawiktionary', 'edit-sowiktionary', 'edit-tiwiktionary', 'edit-sswiktionary',
                           'edit-gnwiktionary', 'edit-iewiktionary', 'edit-pnbwiktionary', 'edit-gdwiktionary',
                           'edit-srwikiquote', 'edit-nowikiquote', 'edit-etwikiquote',
                           'edit-jawikiquote', 'edit-mtwiktionary', 'edit-dvwiktionary', 'edit-iuwiktionary',
                           'edit-kuwikiquote', 'edit-suwiktionary', 'edit-nawiktionary', 'edit-miwiktionary',
                           'edit-roa_rupwiktionary', 'edit-tpiwiktionary', 'edit-gdwiktionary',
                           'edit-lnwiktionary', 'edit-omwiktionary', 'edit-sgwiktionary', 'edit-quwiktionary',
                           'edit-rwwiktionary', 'edit-stwikipedia', 'edit-olowikipedia', 'edit-tnwikipedia',
                           'edit-ffwikipedia', 'edit-dzwikipedia', 'edit-tyvwikipedia', 'edit-dtywikipedia',
                           'edit-xhwikipedia', 'edit-crwikipedia', 'edit-tswikipedia', 'edit-bgwikiquote',
                           'edit-idwikiquote', 'edit-aswikiquote', 'edit-yiwikiquote', 'edit-sawikiquote']
    for index in range(len(train_real_datasets)):
        g_df = pd.read_csv('./data/train/Real/processed/ml_{}.csv'.format(train_real_datasets[index]))
        src_list.append(g_df.u.values)
        dst_list.append(g_df.i.values)
        ts_list.append(g_df.ts.values)

        max_idx = max(g_df.u.values.max(), g_df.i.values.max())
        node_count.append(len(set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))))
        node_list.append(np.unique(np.hstack([src_list[index], dst_list[index]])))
        maxTime_list.append(max(g_df.ts.values))
        adj_list = [[] for _ in range(max_idx + 1)]
        for src, dst, eidx, ts in zip(src_list[index], dst_list[index], g_df.idx.values, ts_list[index]):
            adj_list[dst].append((src, eidx, ts))
        ngh_finder.append(NeighborFinder(adj_list, uniform=UNIFORM))
    return src_list, dst_list, ts_list, node_count, node_list, maxTime_list, ngh_finder


def load_real_true_TKC(dataName):
    g_df = pd.read_csv('./data/test/Real/scores/Result_TKC_{}.csv'.format(dataName))
    test_nodeList = g_df['node_id'].tolist()
    test_nodeList = [int(i) for i in test_nodeList]

    test_tkcList = g_df['tkc_value'].tolist()
    return test_nodeList, test_tkcList


def load_real_train_true_TKC():
    train_nodeList, train_true_tkc = [], []
    train_real_datasets = ['edit-mrwiktionary', 'edit-siwiktionary', 'edit-stwiktionary', 'edit-wowiktionary',
                           'edit-tkwiktionary', 'edit-aywiktionary', 'edit-anwiktionary', 'edit-pawiktionary',
                           'edit-iawiktionary', 'edit-sowiktionary', 'edit-tiwiktionary', 'edit-sswiktionary',
                           'edit-gnwiktionary', 'edit-iewiktionary', 'edit-pnbwiktionary', 'edit-gdwiktionary',
                           'edit-srwikiquote', 'edit-nowikiquote', 'edit-etwikiquote',
                           'edit-jawikiquote', 'edit-mtwiktionary', 'edit-dvwiktionary', 'edit-iuwiktionary',
                           'edit-kuwikiquote', 'edit-suwiktionary', 'edit-nawiktionary', 'edit-miwiktionary',
                           'edit-roa_rupwiktionary', 'edit-tpiwiktionary', 'edit-gdwiktionary',
                           'edit-lnwiktionary', 'edit-omwiktionary', 'edit-sgwiktionary', 'edit-quwiktionary',
                           'edit-rwwiktionary', 'edit-stwikipedia', 'edit-olowikipedia', 'edit-tnwikipedia',
                           'edit-ffwikipedia', 'edit-dzwikipedia', 'edit-tyvwikipedia', 'edit-dtywikipedia',
                           'edit-xhwikipedia', 'edit-crwikipedia', 'edit-tswikipedia', 'edit-bgwikiquote',
                           'edit-idwikiquote', 'edit-aswikiquote', 'edit-yiwikiquote', 'edit-sawikiquote']
    for index in range(len(train_real_datasets)):
        # g_df = pd.read_csv('./data/train/Real/scores/graph_{}_scores.csv'.format(train_real_datasets[index]),
        #                    names=['node_id', 'score'], sep=' ')
        # nodeList = g_df['node_id'].tolist()
        # nodeList = [int(i) for i in nodeList]
        # train_nodeList.append(nodeList)
        #
        # tkcList = g_df['score'].tolist()
        # train_true_tkc.append(tkcList)
        g_df = pd.read_csv('./data/train/Real/scores/Result_TKC_ml_{}.csv'.format(train_real_datasets[index]))
        nodeList = g_df['node_id'].tolist()
        nodeList = [int(i) for i in nodeList]
        train_nodeList.append(nodeList)

        tkcList = g_df['tkc_value'].tolist()
        train_true_tkc.append(tkcList)
    return train_nodeList, train_true_tkc

