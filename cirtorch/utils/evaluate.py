import numpy as np


def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.
    
    Arguments
    ---------
    ranks : zerro-based ranks of positive images
    nres  : number of positive images
    
    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap


# hxq added
def compute_ap_at_k(pos_ranks, mq, k=[100]):
    """
    Computes average precision for given ranked indexes.

    Arguments
    ---------
    pos_ranks : zerro-based ranks of positive images
    mq  : number of positive images
    k : mAP@k

    Returns
    -------
    ap_at_k    : average precision at top k
    """
    pos_ranks_new = pos_ranks + 1
    ap_at_k = np.zeros(len(k))
    for i in range(len(k)):
        # number of images ranked by the system
        nq = len(pos_ranks_new[pos_ranks_new <= k[i]])
        # accumulate trapezoids in PR-plot
        for j in np.arange(nq):
            precision = float(j + 1) / pos_ranks_new[j]
            ap_at_k[i] += (1. / min(mq, k[i])) * precision
    return ap_at_k


def compute_map(ranks, gnd, kappas=[]):
    """
    Computes the mAP for a given set of returned results.

         Usage: 
           map = compute_map (ranks, gnd) 
                 computes mean average precsion (map) only
        
           map, aps, pr, prs = compute_map (ranks, gnd, kappas) 
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query
        
         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    map = 0.
    map_at_k = np.zeros(len(kappas))
    nq = len(gnd) # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0
    # hxq added
    false_neg = []
    false_pos = []

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            # hxq added
            false_pos.append({'img_idxs': np.array([]), 'ranks': np.array([]), 'boundary_num': 0})
            false_neg.append({'img_idxs': np.array([]), 'ranks': np.array([]), 'boundary_num': 0})
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos  = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgndj)]

        ###############################
        # hxq added, find incorrectly identified images
        neg = np.arange(ranks.shape[0])[~np.in1d(ranks[:, i], np.concatenate((qgnd, qgndj)))]
        exclude_junk = np.arange(ranks.shape[0])[~np.in1d(ranks[:, i], qgndj)]
        boundary_num = exclude_junk[len(qgnd)]

        fp_ranks = neg[neg < boundary_num]
        fn_ranks = pos[pos >= boundary_num]

        false_pos.append({'img_idxs': ranks.T[i][fp_ranks], 'ranks': fp_ranks, 'boundary_num': boundary_num})
        false_neg.append({'img_idxs': ranks.T[i][fn_ranks], 'ranks': fn_ranks, 'boundary_num': boundary_num})
        ###############################

        k = 0;
        ij = 0;
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        ap_at_k = compute_ap_at_k(pos, len(qgnd), k=kappas)
        map_at_k += ap_at_k

        # compute precision @ k
        pos += 1 # get it to 1-based
        for j in np.arange(len(kappas)):
            # hxq corrected
            kq = min(max(pos), kappas[j])
            mq = min(len(pos), kappas[j])
            prs[i, j] = (pos <= kq).sum() / mq
            # original
            # kq = min(max(pos), kappas[j]);
            # prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)
    map_at_k = map_at_k / (nq - nempty)

    return map, aps, pr, prs, map_at_k, false_pos, false_neg


def compute_map_and_print(dataset, ranks, gnd, kappas=[1, 5, 10]):

    mismatched_info = []
    map_record = {}

    # old evaluation protocol
    if dataset.startswith('oxford5k') or dataset.startswith('paris6k'):
        map, aps, _, _, map_at_k, fp, fn = compute_map(ranks, gnd)
        mismatched_info.append({'fp': fp, 'fn': fn, 'dataset': dataset})
        print('>> {}: mAP {:.2f}'.format(dataset, np.around(map*100, decimals=2)))
        print('>> {}: mAP@k{} is {}'.format(dataset, kappas, np.around(map_at_k * 100, decimals=5)))

    # new evaluation protocol
    elif dataset.startswith('roxford5k') or dataset.startswith('rparis6k'):
        
        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['easy']])
            g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
            gnd_t.append(g)
        mapE, apsE, mprE, prsE, map_at_k_E, fp, fn = compute_map(ranks, gnd_t, kappas)
        mismatched_info.append({'fp': fp, 'fn': fn, 'dataset': dataset + '_E'})

        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd[i]['junk']])
            gnd_t.append(g)
        mapM, apsM, mprM, prsM, map_at_k_M, fp, fn = compute_map(ranks, gnd_t, kappas)
        mismatched_info.append({'fp': fp, 'fn': fn, 'dataset': dataset + '_M'})

        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
            gnd_t.append(g)
        mapH, apsH, mprH, prsH, map_at_k_H, fp, fn = compute_map(ranks, gnd_t, kappas)
        mismatched_info.append({'fp': fp, 'fn': fn, 'dataset': dataset + '_H'})

        if dataset == 'roxford5k':
            abbr = 'RO'
        elif dataset == 'rparis6k':
            abbr = 'RP'
        else:
            abbr = None
        map_record['{}_E_mAP'.format(abbr)] = mapE * 100
        map_record['{}_M_mAP'.format(abbr)] = mapM * 100
        map_record['{}_H_mAP'.format(abbr)] = mapH * 100
        for i in range(len(map_at_k_E)):
            map_record['{}_E_mAP@{}'.format(abbr, kappas[i])] = map_at_k_E[i] * 100
        for i in range(len(map_at_k_M)):
            map_record['{}_M_mAP@{}'.format(abbr, kappas[i])] = map_at_k_M[i] * 100
        for i in range(len(map_at_k_H)):
            map_record['{}_H_mAP@{}'.format(abbr, kappas[i])] = map_at_k_H[i] * 100
        print('>> {}: mAP E: {}, M: {}, H: {}'.format(dataset, np.around(mapE*100, decimals=2), np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))
        print('>> {}: mAP@k{} E: {}, M: {}, H: {}'.format(dataset, kappas,
                                                         np.around(map_at_k_E * 100, decimals=5),
                                                         np.around(map_at_k_M * 100, decimals=5),
                                                         np.around(map_at_k_H * 100, decimals=5)))
        print('>> {}: mP@k{} E: {}, M: {}, H: {}'.format(dataset, kappas, np.around(mprE*100, decimals=2), np.around(mprM*100, decimals=2), np.around(mprH*100, decimals=2)))

    elif dataset.startswith('google-landmarks-dataset-resize-test'):
        map, aps, mpr, prs, map_at_k, fp, fn = compute_map(ranks, gnd, kappas)
        mismatched_info.append({'fp': fp, 'fn': fn, 'dataset': dataset})
        print('>> {}: mAP {:.5f}'.format(dataset, np.around(map * 100, decimals=5)))
        print('>> {}: mAP@k{} is {}'.format(dataset, kappas, np.around(map_at_k * 100, decimals=5)))
        print('>> {}: mP@k{} is {}'.format(dataset, kappas, np.around(mpr * 100, decimals=5)))
    elif dataset.startswith('google-landmarks-dataset-v2-test'):
        map, aps, mpr, prs, map_at_k, fp, fn = compute_map(ranks, gnd, kappas)
        mismatched_info.append({'fp': fp, 'fn': fn, 'dataset': dataset})
        map_record['GLD2_mAP'] = map * 100
        for i in range(len(map_at_k)):
            map_record['GLD2_mAP@{}'.format(kappas[i])] = map_at_k[i] * 100
        print('>> {}: mAP {:.5f}'.format(dataset, np.around(map * 100, decimals=5)))
        print('>> {}: mAP@k{} is {}'.format(dataset, kappas, np.around(map_at_k * 100, decimals=5)))

    return mismatched_info, map_record

