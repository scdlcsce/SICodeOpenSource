import networkx as nx
import numpy as np
from copy import deepcopy
import itertools
import pickle as pkl
import re
codetype = [ 
    '<operator>.assignment',
    '<operator>.assignmentPlus',
    '<operator>.assignmentMultiplication',
    '<operator>.assignmentDivision',
    '<operator>.assignmentMinus',
    '<operators>.assignmentAnd',
    '<operator>.assignmentAnd',
    '<operators>.assignmentOr',
    '<operators>.assignmentXor',
    '<operators>.assignmentModulo',
    '<operators>.assignmentExponentiation',
    '<operators>.assignmentArithmeticShiftRight',
    '<operators>.LogicalShifRight',
    '<operators>.assignmentShiftLeft',
    '<operator>.logicalNot',
    '<operator>.logicalOr',
    '<operator>.logicalAnd',
    '<operator>.equals',
    '<operator>.notEquals',
    '<operator>.indirectMemberAccess',
    '<operator>.computedMemberAccess',
    '<operator>.addressOf',
    '<operator>.cast',
    '<operator>.conditionalExpression',
    '<operator>.postIncrement',
    '<operator>.preIncrement',
    '<operator>.postDecrement',
    '<operator>.preDecrement',
    '<operator>.or',
    '<operator>.and',
    '<operator>.addition',
    '<operator>.subtraction',
    '<operator>.multiplication',
    '<operator>.division',
    '<operator>.modulo',
    '<operator>.minus', #drop
    '<operator>.plus',
    '<operator>.not',
    '<operator>.arithmeticShiftRight',
    '<operator>.arithmeticShiftLeft',
    '<operator>.lessEqualsThan',
    '<operator>.greaterEqualsThan',
    '<operator>.lessThan',
    '<operator>.greaterThan',
    '<operator>.indirection',
    '<operator>.memberAccess',
    '<operator>.sizeOf',
    '<operator>.shiftLeft',
    '<operator>.shiftRight']

assigntype = {
    '<operator>.assignmentPlus': ['<operator>.plus', '<operator>.assignment'],
    '<operator>.assignmentMultiplication': ['<operator>.multiplication','<operator>.assignment'],
    '<operator>.assignmentDivision':['<operator>.division','<operator>.assignment'],
    '<operator>.assignmentMinus':['<operator>.subtraction','<operator>.assignment'],
    '<operators>.assignmentAnd':['<operator>.and','<operator>.assignment'],
    '<operator>.assignmentAnd':['<operator>.and','<operator>.assignment'],
    '<operators>.assignmentOr':['<operator>.or','<operator>.assignment'],
    '<operators>.assignmentXor':['<operator>.or','<operator>.assignment'],
    '<operators>.assignmentModulo':['<operator>.modulo','<operator>.assignment'],
    '<operators>.assignmentExponentiation':['<operator>.multiplication','<operator>.assignment'],
    '<operators>.assignmentArithmeticShiftRight':['<operator>.arithmeticShiftRight','<operator>.assignment'],
    '<operators>.assignmentShiftLeft': ['<operators>.LogicalShifRight','<operator>.assignment'],
    '<operator>.postIncrement':['<operator>.assignment', '<operator>.plus'],
    '<operator>.preIncrement':['<operator>.plus', '<operator>.assignment'],
    '<operator>.postDecrement':['<operator>.assignment','<operator>.subtraction'],
    '<operator>.preDecrement':['<operator>.subtraction', '<operator>.assignment']
    }

code2type = { 
    '<operator>.plus':                          1,
    '<operator>.addition':                      1,
    '<operator>.subtraction':                   2,
    '<operator>.multiplication':                3,
    '<operator>.division':                      4,
    '<operator>.modulo':                        5,
    '<operator>.or':                            6,
    '<operator>.and':                           7,
    '<operator>.not':                           8,
    '<operator>.shiftRight':                    9,
    '<operators>.LogicalShifRight':             9,
    '<operator>.arithmeticShiftRight':          9,
    '<operator>.arithmeticShiftLeft':           9,
    '<operator>.shiftLeft':                     9,
    '<operator>.equals':                        10,
    '<operator>.notEquals':                     11,
    '<operator>.lessEqualsThan':                12,
    '<operator>.lessThan':                      12,
    '<operator>.greaterEqualsThan':             13,
    '<operator>.greaterThan':                   13,
    '<operator>.logicalNot':                    14,
    '<operator>.logicalOr':                     15,
    '<operator>.logicalAnd':                    16,
    '<operator>.sizeOf':                        17,
    '<operator>.addressOf':                     18,
    # --- drop ---
    '<operator>.minus':                         19, 
    '<operator>.cast':                          20,
    '<operator>.indirectMemberAccess':          21,          
    '<operator>.computedMemberAccess':          21,
    '<operator>.indirection':                   21,
    '<operator>.memberAccess':                  21,
    '<operator>.assignment':                    22,


}
    # 76+ 22 + 1 + 1
    

def getgraph(funcname, funcpdg):
    g = nx.DiGraph()
    g.name = funcname
    for c in funcpdg:
        prop = c['properties']
        propdict = dict()
        for pp in prop:
            propdict[pp['key']] = pp['value']
        
        g.add_node(c['id'].split('.')[-1], property = propdict)

        for e in c['edges']:
            try:
                t = g.edges()[(e['out'].split('.')[-1], e['in'].split('.')[-1])]['type']
            except Exception:
                t = []
            if 'Cfg@' in e['id']:
                t.append('cfg')
            elif 'ReachingDef@' in e['id']:
                t.append('ddg')
            elif 'Cdg@' in e['id']:
                t.append('cdg')
            else:
                t.append('unknown')
            g.add_edge(e['out'].split('.')[-1], e['in'].split('.')[-1], type=t)

    for e in list(g.edges()):
        g.edges()[e]['type'] = list(set(g.edges()[e]['type']))

    for n in list(g.nodes()):
        try:
            pdict = g.nodes()[n]['property']
        except:
            g.nodes()[n]['line'] = []
            g.nodes()[n]['type'] = []
            g.nodes()[n]['stm'] = ''
            continue
        g.nodes()[n]['line'] = pdict['LINE_NUMBER']
        if 'Return@' in n:
            g.nodes()[n]['type'] = 'return'
            g.nodes()[n]['stm'] = 'return'
            g.nodes()[n].pop('property')
            continue
        if 'Unknown@' in n:
            g.nodes()[n]['type'] = 'unknown'
            g.nodes()[n]['stm'] = 'unknown'
            g.nodes()[n].pop('property')
            continue
        if 'Identifier@' in n:
            g.nodes()[n]['type'] = 'identifier'
            g.nodes()[n]['stm'] = pdict['CODE'].replace('\"','\\\"')
            g.nodes()[n].pop('property')
            continue
        if 'Literal@' in n:
            g.nodes()[n]['type'] = 'literal'
            g.nodes()[n]['stm'] = pdict['CODE'].replace('\"','\\\"')
            g.nodes()[n].pop('property')
            continue
        try:
            stm = pdict['NAME']
        except Exception:
            stm = ''
        if stm == '':
            g.nodes()[n]['type'] = 'block'
            g.nodes()[n]['stm'] = ''
        elif not stm in codetype:
            g.nodes()[n]['type'] = 'api'
            fname = stm.lower().strip()
            if '()' in fname:
                fname = fname.replace('()','')
            if '->' in fname:
                fname = fname.split('->')[-1]
            if '.' in fname:
                fname = fname.split('.')[-1]
            if ')' in fname:
                fname = fname.replace(')','')
            if '(' in fname:
                fname = fname.split('(')[0]
            if '~' in fname:
                fname = fname.split('~')[1]
            if ',' in fname:
                fname = fname.split(',')[0].strip()
            if '[' in fname:
                fname = fname.split('[')[0].strip()
            g.nodes()[n]['stm'] = fname
        else:
            g.nodes()[n]['type'] = 'opr'
            g.nodes()[n]['stm'] = stm
        g.nodes()[n].pop('property')
    return g

def removeid(g):
    cfg = nx.DiGraph()
    cfg.name = g.name
    nlist = list(g.nodes())
    for n in nlist:
        cfg.add_node(n, stm = g.nodes()[n]['stm'], type = g.nodes()[n]['type'], line = g.nodes()[n]['line'])
    cfg.add_edges_from([ e for e in g.edges() if 'cfg' in g.edges()[e]['type']])
    nlist = list(cfg.nodes())
    for n in nlist:
        if 'Unknown@' in n:
            pre = [p for p in cfg.predecessors(n)]
            suc = [s for s in cfg.successors(n)]
            for p in pre:
                for s in suc:
                    cfg.add_edge(p,s)
            cfg.remove_node(n)
        elif not ('Call@' in n or 'Return@' in n or 'Method@' in n):
            pre = [p for p in cfg.predecessors(n)]
            suc = [s for s in cfg.successors(n)]
            call_suc = [s for s in suc if ('Call@' in s or 'Return@' in s or 'Method@' in s)]
            if len(call_suc) >1:
                cfg.add_node(n, stm = '<operator>.equals', type = 'opr', line = g.nodes()[n]['line'])
            else:
                if len(pre)>0 and len(suc) >0:
                    for p in pre:
                        for s in suc:
                            cfg.add_edge(p,s)
                cfg.remove_node(n)
    for e in list(cfg.edges()):
        cfg.edges()[e]['type'] = ['cfg']
    nlist = list(cfg.nodes())
    for e in list(g.edges()):
        if not (e[0] in nlist and e[1] in nlist):
            continue
        try:
            t = cfg.edges()[e]['type']
        except Exception:
            t = []
        if 'ddg' in g.edges()[e]['type']:
            t.append('ddg')
        if 'cdg' in g.edges()[e]['type']:
            t.append('cdg')
        cfg.add_edge(e[0], e[1], type = t)

    for n in nlist:
        if 'Call' in n or 'Return' in n or 'Method@' in n:
            continue
        if 'Identifier' in n:
            n_name = 'i_Call@' + n.split('@')[1]
        elif 'Literal' in n:
            n_name = 'l_Call@' + n.split('@')[1]
        
        cfg.add_node(n_name, stm = cfg.nodes()[n]['stm'], type = cfg.nodes()[n]['type'], line = cfg.nodes()[n]['line'])
        suc = list(cfg.successors(n))
        pre = list(cfg.predecessors(n))
            
        for p in pre:
            cfg.add_edge(p,n_name, type = cfg.edges()[(p,n)]['type'])
        for s in suc:
            cfg.add_edge(n_name, s, type = cfg.edges()[(n,s)]['type'])
        cfg.remove_node(n)
    return cfg


def normgraph(g):
    nlist = list(g.nodes())
    for n in nlist:
        
        if g.nodes()[n]['stm'] == '<operator>.conditionalExpression':
            g.nodes()[n]['stm'] = '<operator>.assignment'
            continue

        if g.nodes()[n]['stm'] in assigntype:
            stm1, stm2 = assigntype[g.nodes()[n]['stm']]
            l = g.nodes()[n]['line']
            t = 'opr'
            n1 = n+'_p'
            n2 = n+'_s'
            g.add_node(n1, line = l,type = t,stm = stm1)
            g.add_node(n2, line = l,type = t,stm = stm2)
            g.add_edge(n1, n2, type = ['cfg','ddg'])
            pre = list(g.predecessors(n))
            suc = list(g.successors(n))
            for p in pre:
                g.add_edge(p,n1, type = g.edges()[(p, n)]['type'])
            for s in suc:
                g.add_edge(n2,s, type = g.edges()[(n, s)]['type'])
            g.remove_node(n)

    return g


def removeassign(g):
    # cfg = nx.DiGraph()
    # cfg.add_nodes_from(g.nodes())
    # cfg.add_edges_from([ e for e in g.edges() if 'cfg' in g.edges()[e]['type']])

    for n in list(g.nodes()):
        if g.nodes()[n]['stm'] in['<operator>.indirectMemberAccess','<operator>.computedMemberAccess','<operator>.indirection','<operator>.memberAccess', '<operator>.cast' ,'<operator>.minus']:
            cfg_pre = [p for p in list(g.predecessors(n)) if 'cfg' in g.edges()[(p,n)]['type']]
            cdg_pre = [p for p in list(g.predecessors(n)) if 'cdg' in g.edges()[(p,n)]['type']]
            ddg_pre = [p for p in list(g.predecessors(n)) if 'ddg' in g.edges()[(p,n)]['type']]
            cfg_suc = [s for s in list(g.successors(n)) if 'cfg' in g.edges()[(n,s)]['type']]
            cdg_suc = [s for s in list(g.successors(n)) if 'cdg' in g.edges()[(n,s)]['type']]
            ddg_suc = [s for s in list(g.successors(n)) if 'ddg' in g.edges()[(n,s)]['type']]
            
            if len(cfg_suc) > 1:
                g.nodes()[n]['stm'] = '<operator>.equals'
                continue
            if len(cfg_suc) == 0:
                g.remove_node(n)
                continue
            cfg_suc = cfg_suc[0]
            for p in cfg_pre:
                try:
                    t = list(set(g.edges()[(p, cfg_suc)]['type'] + ['cfg']))
                    g.edges()[(p, cfg_suc)]['type'] = t
                except Exception:
                    g.add_edge(p, cfg_suc, type = ['cfg'])

            for p in ddg_pre:
                try:
                    t = list(set(g.edges()[(p, cfg_suc)]['type'] + ['ddg']))
                    g.edges()[(p, cfg_suc)]['type'] = t
                except Exception:
                    g.add_edge(p, cfg_suc, type = ['ddg'])

            for s in ddg_suc:
                try:
                    t = list(set(g.edges()[(cfg_suc, s)]['type'] + ['ddg']))
                    g.edges()[(cfg_suc, s)]['type'] = t
                except Exception:
                    g.add_edge(cfg_suc,s, type = ['ddg'])

            g.remove_node(n)
            continue


    nlist = {n: int(g.nodes()[n]['line']) for n in list(g.nodes()) if g.nodes()[n]['stm'] == '<operator>.assignment'}
    nlist = [n[0] for n in sorted(nlist.items(), key=lambda d: d[1], reverse=True)]
    for n in nlist:
        cfg_prenode = [p for p in list(g.predecessors(n)) if 'cfg' in g.edges()[(p, n)]['type'] ]
        cfg_sucnode = [s for s in list(g.successors(n)) if 'cfg' in g.edges()[(n, s)]['type'] ]
        ddg_sucnode = [s for s in list(g.successors(n)) if 'ddg' in g.edges()[(n, s)]['type'] ]
        cdg_sucnode = [s for s in list(g.successors(n)) if 'cdg' in g.edges()[(n, s)]['type'] ]
        if len(cfg_sucnode) > 1:
            g.nodes()[n]['stm'] = '<operator>.equals'
            continue
        if len(cfg_sucnode) == 0:
            g.remove_node(n)
            continue
        cfg_sucnode = cfg_sucnode[0]

        for p in cfg_prenode:
            if (p,cfg_sucnode) in g.edges():
                g.edges()[(p,cfg_sucnode)]['type'] = list(set(g.edges()[(p,cfg_sucnode)]['type'] + ['cfg']))
            else:
                g.add_edge(p,cfg_sucnode,type = ['cfg'])

        sameline_pre = [p for p in cfg_prenode if g.nodes()[p]['line'] == g.nodes()[n]['line']]

        if len(sameline_pre) > 0:
            sameline_pre = sameline_pre[0]
            for s in ddg_sucnode:
                if (sameline_pre,s) in g.edges():
                    g.edges()[(sameline_pre,s)]['type'] = list(set(g.edges()[(sameline_pre,s)]['type'] + ['ddg']))
                else:
                    g.add_edge(sameline_pre,s,type = ['ddg'])

            for s in cdg_sucnode:
                if (sameline_pre,s) in g.edges():
                    g.edges()[(sameline_pre,s)]['type'] = list(set(g.edges()[(sameline_pre,s)]['type'] + ['cdg']))
                else:
                    g.add_edge(sameline_pre,s,type = ['cdg'])
        g.remove_node(n)
    return g


def graphfork(g):
    for e in list(g.edges()):
        if e[0] == e[1]:
            g.remove_edge(e[0], e[1])

    for n in g.nodes():
        succs = g.successors(n)
        s = [s for s in succs if 'cfg' in g.edges()[(n,s)]['type']]
        pres = g.predecessors(n)
        p = [p for p in pres if 'cfg' in g.edges()[(p,n)]['type']]

        if len(s) >1 and len(p) > 1:
            g.add_node(n, stm = g.nodes()[n]['stm'], type = g.nodes()[n]['type'], line = g.nodes()[n]['line'], fork = True, unfork = True)
        elif len(s) > 1:
            g.add_node(n, stm = g.nodes()[n]['stm'], type = g.nodes()[n]['type'], line = g.nodes()[n]['line'], fork = True, unfork = False)
        elif len(p) > 1:
            g.add_node(n, stm = g.nodes()[n]['stm'], type = g.nodes()[n]['type'], line = g.nodes()[n]['line'], fork = False, unfork = True)
        else:
            g.add_node(n, stm = g.nodes()[n]['stm'], type = g.nodes()[n]['type'], line = g.nodes()[n]['line'], fork = False, unfork = False)
    return g


def forward(bbg, n, hopsize, edge_type):
    count = 0
    sucedge  = []
    while count < hopsize :
        count += 1
        suc = []
        for ni in n:
            sucnode = list(bbg.successors(ni))
            sucnode = [s for s in sucnode if edge_type in bbg.edges()[(ni, s)]['type'] ]
            suc += sucnode
            sucedge += [(ni, s) for s in sucnode]
        newnode = []
        for s in suc:
           if not s in n:
                newnode.append(s)
        if len(newnode) == 0:
            break
        n += newnode
    return n, sucedge


def backward(bbg, n, hopsize, edge_type):
    count = 0
    preedge  = []
    while count < hopsize:
        count += 1
        pre = []
        for ni in n:
            prenode = list(bbg.predecessors(ni))
            prenode = [p for p in prenode if edge_type in bbg.edges()[(p, ni)]['type'] ]
            pre += prenode
            preedge += [(p, ni) for p in prenode]
        newnode = []
        for p in pre:
            if not p in n:
                newnode.append(p)
        if len(newnode) == 0:
            break
        n += newnode
    return n, preedge

def g_sample(tgraph, n, hopsize):

    nsus_cfg, edgesus_cfg = forward(tgraph, [n], hopsize, 'cfg')
    npre_cfg, edgepre_cfg = backward(tgraph, [n], hopsize, 'cfg')

    # nsus_cdg, edgesus_cdg = forward(tgraph, [n], hopsize, 'cdg')
    # npre_cdg, edgepre_cdg = backward(tgraph, [n], hopsize, 'cdg')

    nsus_ddg, edgesus_ddg = forward(tgraph, [n], hopsize, 'ddg')
    npre_ddg, edgepre_ddg = backward(tgraph, [n], hopsize, 'ddg')

    g = nx.DiGraph()
    # nodes = [n] + list(set(nsus_cfg + npre_cfg+ nsus_cdg + npre_cdg + nsus_ddg + npre_ddg))
    nodes = [n] + list(set(nsus_cfg + npre_cfg+ nsus_ddg + npre_ddg))

    if len(nodes) > 110: #
        return None # 

    for n in nodes:
        g.add_node(n, line = tgraph.nodes()[n]['line'], type = tgraph.nodes()[n]['type'], stm = tgraph.nodes()[n]['stm'])

    edgetype = {}
    for e in edgesus_cfg:
        if e in edgetype:
            edgetype[e] = list(set(edgetype[e] + ['cfg']))
        else:
            edgetype[e] = ['cfg']
    for e in edgepre_cfg:
        if e in edgetype:
            edgetype[e] = list(set(edgetype[e] + ['cfg']))
        else:
            edgetype[e] = ['cfg']

    for e in edgesus_ddg:
        if e in edgetype:
            edgetype[e] = list(set(edgetype[e] + ['ddg']))
        else:
            edgetype[e] = ['ddg']
    for e in edgepre_ddg:
        if e in edgetype:
            edgetype[e] = list(set(edgetype[e] + ['ddg']))
        else:
            edgetype[e] = ['ddg']

    # for e in edgesus_cdg:
    #     if e in edgetype:
    #         edgetype[e] = list(set(edgetype[e] + ['cdg']))
    #     else:
    #         edgetype[e] = ['cdg']
    # for e in edgepre_cdg:
    #     if e in edgetype:
    #         edgetype[e] = list(set(edgetype[e] + ['cdg']))
    #     else:
    #         edgetype[e] = ['cdg']
        
    for ep in edgetype:
        g.add_edge(ep[0], ep[1], type = edgetype[e])

    return g


# def g_sample_tmp(tgraph, n_cent, hopsize):
#     count = 0
#     sucedge  = []
#     n1 = [n_cent]
#     n2 = [n_cent]
#     while count < hopsize :
#         count += 1
#         suc = []
#         for ni in n1:
#             sucnode = list(tgraph.successors(ni))
#             suc += sucnode
#             sucedge += [(ni, s) for s in sucnode]
#         newnode = []
#         for s in suc:
#            if not s in n1:
#                 newnode.append(s)
#         if len(newnode) == 0:
#             break
#         n1 += newnode
#     nsus_cfg, edgesus_cfg = n1, sucedge
#     count = 0
#     preedge  = []
#     while count < hopsize:
#         count += 1
#         pre = []
#         for ni in n2:
#             prenode = list(tgraph.predecessors(ni))
#             pre += prenode
#             preedge += [(p, ni) for p in prenode]
#         newnode = []
#         for p in pre:
#             if not p in n2:
#                 newnode.append(p)
#         if len(newnode) == 0:
#             break
#         n2 += newnode
#     npre_cfg, edgepre_cfg = n2, preedge
#     g = nx.Graph()
#     nodes = [n_cent] + list(set(nsus_cfg + npre_cfg))
#     for n in nodes:
#         g.add_node(n, line = tgraph.nodes()[n]['line'], type = tgraph.nodes()[n]['type'], stm = tgraph.nodes()[n]['stm'])

#     for e in edgesus_cfg:
#         g.add_edge(e[0], e[1], type = tgraph.edges()[e]['type'])
#     for e in edgepre_cfg:
#         g.add_edge(e[0], e[1], type = tgraph.edges()[e]['type'])
#     return g

def getnodeblock(ast, clayer):
    blocks = []
    for a in ast:
        if len(a) == 0:
            continue
        if isinstance(a, list):
            alayer = {l.split('@')[0]:int(l.split('@')[1]) for l in str(a).replace(']',' ').split('\'') if '@' in l}
            if min(list(alayer.values())) > clayer:
                blocks.append(a)
            elif max(list(alayer.values())) <= clayer:
                continue
            else:
                blocks += getnodeblock(a, clayer)
    return blocks


def g_sample_group(tgraph, n_cent, hopsize, ast):
    gsample = g_sample(tgraph, n_cent, hopsize)
    if gsample == None:
        return None
    if not len(ast) == 0:
        # g = deepcopy(tgraph)
        nodeline = [{n : gsample.nodes()[n]['line']} for n in gsample.nodes()]
        linenode = itertools.groupby(nodeline, lambda x:list(x.items())[0][1])
        line2node = {i[0]:[list(j.keys())[0] for j in i[1]] for i in linenode}
        centline = tgraph.nodes()[n_cent]['line']
        layers = {l.split('@')[0]: l.split('@')[1] for l in str(ast).replace(']',' ').split('\'') if '@' in l}
        centerlayer = int(layers[centline])
        blocks = getnodeblock(ast, centerlayer)

        if blocks == []:
            return gsample

        nodeblocks = []
        for b in blocks:
            s = str(b)
            blockre = re.findall(r'\'[0-9]*\@[0-9]*\'', s)
            bline = [bi[1:-1].split('@')[0] for bi in blockre]
            bnode = []
            for l in bline:
                try:
                    bnode += line2node[l]
                except Exception:
                    pass
            bnode = list(set(bnode))
            if len(bnode)>1:
                nodeblocks.append(bnode)
        if nodeblocks == []:
            return gsample    
        for bidx,bnode in enumerate(nodeblocks):
            bnode_remain = [b for b in bnode if b in list(gsample.nodes())]
            if len(bnode_remain) <= 1:
                continue
            cfg_pres = []
            cfg_sucs = []
            ddg_pres = []
            ddg_sucs = []
            cdg_pres = []
            cdg_sucs = []
            for n in bnode_remain:
                cfg_pres += [p for p in list(gsample.predecessors(n)) if 'cfg' in gsample.edges()[(p,n)]['type']]
                # cdg_pres += [p for p in list(g.predecessors(n)) if 'cdg' in g.edges()[(p,n)]['type']]
                ddg_pres += [p for p in list(gsample.predecessors(n)) if 'ddg' in gsample.edges()[(p,n)]['type']]
                cfg_sucs += [s for s in list(gsample.successors(n)) if 'cfg' in gsample.edges()[(n,s)]['type']]
                # cdg_sucs += [s for s in list(g.successors(n)) if 'cdg' in g.edges()[(n,s)]['type']]
                ddg_sucs += [s for s in list(gsample.successors(n)) if 'ddg' in gsample.edges()[(n,s)]['type']]
            cfg_pres = [n for n in cfg_pres if not n in bnode]
            cfg_sucs = [n for n in cfg_sucs if not n in bnode]
            ddg_pres = [n for n in ddg_pres if not n in bnode]
            ddg_sucs = [n for n in ddg_sucs if not n in bnode]
            # cdg_pres = [n for n in cdg_pres if not n in bnode]
            # cdg_sucs = [n for n in cdg_sucs if not n in bnode]
            blockname = 'block'+str(bidx)
            gsample.add_node(blockname, type = 'block', stm = 'block',  line = '-1')
            for p in list(set(cfg_pres + ddg_pres + cdg_pres)):
                t = ['cfg' for pi in [p] if pi in cfg_pres] + ['ddg' for pi in [p] if pi in ddg_pres] + ['cdg' for pi in [p] if pi in cdg_pres]
                gsample.add_edge(p, blockname, type = t)
            for s in list(set(cfg_sucs + ddg_sucs + cdg_sucs)):
                t = ['cfg' for si in [s] if si in cfg_sucs] + ['ddg' for si in [s] if si in ddg_sucs] + ['cdg' for si in [s] if si in cdg_sucs]
                gsample.add_edge(blockname, s, type = t)
            for bn in bnode_remain:
                gsample.remove_node(bn)
    return gsample


def gtostr(g, n):
    gnode = [n] + [successor for successors in dict(nx.bfs_successors(g, n)).values() for successor in successors]
    adj = []
    types = [g.nodes()[gn]['type'] for gn in g.nodes()]
    gedge = list(g.edges())
    for idxi, i in enumerate(gnode):
        for idxj, j in enumerate(gnode):
            if (i,j) in gedge:
                adj.append(str([idxi,idxj]))
                adj.append(str([idxj,idxi]))
    return [len(types), str([types, sorted(list(set(adj)))])]


def g2feats(g, ldict, input_dim, wvec_dim):
    feat = np.zeros(input_dim)
    gnode = list(g.nodes())
    for n in gnode:
        callstm = g.nodes()[n]["stm"]
        calltype = g.nodes()[n]["type"]
        if calltype == 'api':
            label = ldict[callstm]
            gvec = label
        elif calltype == 'opr':
            try:
                label = code2type[callstm]
                gvec = wvec_dim + label - 1
            except Exception:
                print(callstm)
                gvec = input_dim-1
        else: 
            gvec = input_dim-1
        feat[gvec] += 1
        g.nodes()[n]['label'] = gvec
    return feat, g

def seedg2feat(gnode, gedges, tgraph, ldict, input_dim, wvec_dim, iline):
    sg = nx.Graph()
    feat = np.zeros(input_dim)
    for idxn, n in enumerate(gnode):
        gvec = wvec_dim
        try:
            callstm = tgraph.nodes()[n]["stm"]
            calltype = tgraph.nodes()[n]["type"]
            if calltype == 'api':
                label = ldict[callstm]
                gvec = label - 1
            elif calltype == 'opr':
                label = code2type[callstm]
                gvec = wvec_dim + label
            else: 
                gvec = input_dim-1
        except KeyError:
            pass
        sg.add_node('n'+str(idxn), label = gvec)
        
    for e in gedges:
        if e[0] in gnode and e[1] in gnode:
            sg.add_edge('n'+str(gnode.index(e[0])), 'n'+str(gnode.index(e[1])))
    for n in list(sg.nodes()):
        f = sg.nodes()[n]['label']
        feat[f] += 1
    return sg, feat


def seed2g(tgraph, ilines, centapi, hopsize):
    centnode = None
    for n in tgraph.nodes():
        try:
            if int(tgraph.nodes()[n]['line']) == ilines[0] and tgraph.nodes()[n]['stm'] == centapi:
                centnode = n
        except Exception:
            pass
    if centnode == None:
        return None, None

    nsus_cfg, edgesus_cfg = forward(tgraph, [centnode], hopsize, 'cfg')
    npre_cfg, edgepre_cfg = backward(tgraph, [centnode], hopsize, 'cfg')

    nsus_ddg, edgesus_ddg = forward(tgraph, [centnode], hopsize, 'ddg')
    npre_ddg, edgepre_ddg = backward(tgraph, [centnode], hopsize, 'ddg')

    g = nx.DiGraph()
    nodes = [centnode] + list(set(nsus_cfg + npre_cfg+ nsus_ddg + npre_ddg))

    for n in nodes:
        l = int(tgraph.nodes()[n]['line'])
        if l in ilines:
            g.add_node(n, line = tgraph.nodes()[n]['line'], type = tgraph.nodes()[n]['type'], stm = tgraph.nodes()[n]['stm'])

    edgetype = {}
    for e in edgesus_cfg:
        if e in edgetype:
            edgetype[e] = list(set(edgetype[e] + ['cfg']))
        else:
            edgetype[e] = ['cfg']
    for e in edgepre_cfg:
        if e in edgetype:
            edgetype[e] = list(set(edgetype[e] + ['cfg']))
        else:
            edgetype[e] = ['cfg']

    for e in edgesus_ddg:
        if e in edgetype:
            edgetype[e] = list(set(edgetype[e] + ['ddg']))
        else:
            edgetype[e] = ['ddg']
    for e in edgepre_ddg:
        if e in edgetype:
            edgetype[e] = list(set(edgetype[e] + ['ddg']))
        else:
            edgetype[e] = ['ddg']
    for ep in edgetype:
        if ep[0] in g.nodes() and ep[1] in g.nodes():
            g.add_edge(ep[0], ep[1], type = edgetype[e])
    return centnode, g


def group_seed2g(tgraph, ilines, centapi, hopsize, ast):
    
    centnode, gsample = seed2g(tgraph, ilines, centapi, hopsize)
    
    if not len(ast) == 0:
        nodeline = [{n : gsample.nodes()[n]['line']} for n in gsample.nodes()]
        linenode = itertools.groupby(nodeline, lambda x:list(x.items())[0][1])
        line2node = {i[0]:[list(j.keys())[0] for j in i[1]] for i in linenode}
        centline = tgraph.nodes()[centnode]['line']
        layers = {l.split('@')[0]: l.split('@')[1] for l in str(ast).replace(']',' ').split('\'') if '@' in l}
        centerlayer = int(layers[centline])
        blocks = getnodeblock(ast, centerlayer)
        if blocks == []:
            return centnode, gsample
        nodeblocks = []
        for b in blocks:
            bnode = []
            for bi in b:
                try:
                    l = int(bi.split('@')[0])
                    bnode += line2node[str(l)]
                except Exception:
                    pass
            bnode = list(set(bnode))
            if len(bnode)>1:
                nodeblocks.append(bnode)
        if nodeblocks == []:
            return centnode, gsample 
        for bidx,bnode in enumerate(nodeblocks):
            bnode_remain = [b for b in bnode if b in list(gsample.nodes())]
            if len(bnode_remain) <= 1:
                continue
            cfg_pres = []
            cfg_sucs = []
            ddg_pres = []
            ddg_sucs = []
            cdg_pres = []
            cdg_sucs = []
            for n in bnode_remain:
                cfg_pres += [p for p in list(gsample.predecessors(n)) if 'cfg' in gsample.edges()[(p,n)]['type']]
                # cdg_pres += [p for p in list(g.predecessors(n)) if 'cdg' in g.edges()[(p,n)]['type']]
                # ddg_pres += [p for p in list(g.predecessors(n)) if 'ddg' in g.edges()[(p,n)]['type']]
                cfg_sucs += [s for s in list(gsample.successors(n)) if 'cfg' in gsample.edges()[(n,s)]['type']]
                # cdg_sucs += [s for s in list(g.successors(n)) if 'cdg' in g.edges()[(n,s)]['type']]
                # ddg_sucs += [s for s in list(g.successors(n)) if 'ddg' in g.edges()[(n,s)]['type']]
            cfg_pres = [n for n in cfg_pres if not n in bnode]
            cfg_sucs = [n for n in cfg_sucs if not n in bnode]
            # ddg_pres = [n for n in ddg_pres if not n in bnode]
            # ddg_sucs = [n for n in ddg_sucs if not n in bnode]
            # cdg_pres = [n for n in cdg_pres if not n in bnode]
            # cdg_sucs = [n for n in cdg_sucs if not n in bnode]
            blockname = 'block'+str(bidx)
            gsample.add_node(blockname, type = 'block', stm = 'block',  line = '-1')
            for p in list(set(cfg_pres + ddg_pres + cdg_pres)):
                t = ['cfg' for pi in [p] if pi in cfg_pres] + ['ddg' for pi in [p] if pi in ddg_pres] + ['cdg' for pi in [p] if pi in cdg_pres]
                gsample.add_edge(p, blockname, type = t)
            for s in list(set(cfg_sucs + ddg_sucs + cdg_sucs)):
                t = ['cfg' for si in [s] if si in cfg_sucs] + ['ddg' for si in [s] if si in ddg_sucs] + ['cdg' for si in [s] if si in cdg_sucs]
                gsample.add_edge(blockname, s, type = t)
            for bn in bnode_remain:
                gsample.remove_node(bn)

    return centnode, gsample

def getcorpus(funcpdg, pdg):
    linedict = {}
    for n in list(pdg.nodes()):
        if pdg.nodes()[n]['type'] == 'api':
            l = int(pdg.nodes()[n]['line'])
            if not l in linedict:
                linedict[l] = [n]
            else:
                linedict[l] = linedict[l] + [n]

    for k in list(linedict.keys()):
        if len(linedict[k]) == 1:
            continue
        else:
            columndict = {}
            for c in funcpdg:
                n = c['id'].split('.')[-1]
                if n in linedict[k]:
                    prop = c['properties']  
                    propdict = dict()
                    for pp in prop:
                        propdict[pp['key']] = pp['value']
                    if 'COLUMN_NUMBER' in propdict:
                        columndict[int(propdict['COLUMN_NUMBER'])] = n
                    else:
                        columndict[-1] = n
            columndict = [columndict[l] for l in sorted(columndict.keys(), reverse=True)]
            linedict[k] = columndict
    codelist = []
    for l in sorted(linedict.keys()):
        codelist += linedict[l]
    c = ' '.join([pdg.nodes()[n]['stm'].strip() for n in codelist])
    return c 

        