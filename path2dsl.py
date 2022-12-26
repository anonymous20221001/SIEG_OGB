# -*- coding: UTF-8 -*-
import json
import math
import pdb

'''
GraphStructure {
  A [MKG.PartnerID]
  B [MKG.MifShop]
  C [MKG.AppService]
  D [MKG.KoubeiShop]
  
  B->A [hasPID] as e1
  C->B [relatedShop] as e2
  C->D [relatedShop] as e3
}
Rule {
}
Action {
  distinctGet(A._s,D._s)
}
'''

class PathFeatureToDSLUDF(object):

    def evaluate(self, path_feature, path_name, mark=''):
        path_features = path_feature.strip().split(';')
        relation_list = path_features[0].split('>>>')
        if len(relation_list) == 0:
            return ''

        graph_struct_paragraph1 = ''
        graph_struct_paragraph2 = ''
        head = ''
        tail = ''
        for i in range(len(relation_list)):
            relation = relation_list[i]
            pos = relation.find('_inv')
            array = ''
            pdb.set_trace()
            if pos > 0 and pos + 4 == len(relation):
                relation = relation[0:pos]
                [o, p, s] = relation.split('_')
                #  array = '<-'
                array = '{}->{}'.format(chr(ord('A')+i+1), chr(ord('A')+i))
            else:
                [s, p, o] = relation.split('_')
                #  array = '->'
                array = '{}->{}'.format(chr(ord('A')+i), chr(ord('A')+i+1))
            # print('{}: {}->{}->{}'.format(relation_list[i], s, p, o))

            if i == 0:
                if mark.lower() == 'head':
                    graph_struct_paragraph1 = "  {} [{},__start__='true']\n".format(chr(ord('A')+i), s)
                else:
                    graph_struct_paragraph1 = "  {} [{}]\n".format(chr(ord('A')+i), s)
                head = chr(ord('A')+i)
            if i == len(relation_list) - 1:
                tail = chr(ord('A')+i+1)
            if i == len(relation_list) - 1 and mark.lower() == 'tail':
                graph_struct_paragraph1 = graph_struct_paragraph1 + "  {} [{},__start__='true']\n".format(chr(ord('A')+i+1), o)
            else:
                graph_struct_paragraph1 = graph_struct_paragraph1 + "  {} [{}]\n".format(chr(ord('A')+i+1), o)
            # graph_struct_paragraph2 = graph_struct_paragraph2 + "  {}{}{} [{}] as e{}\n'.format(chr(ord('A')+i), array, chr(ord('A')+i+1), p, i+1)
            graph_struct_paragraph2 = graph_struct_paragraph2 + '  {} [{}] as e{}\n'.format(array, p, i+1)

        dsl_str = 'GraphStructure {{\n{}\n{}}}\n'.format(graph_struct_paragraph1, graph_struct_paragraph2)
        dsl_str = dsl_str + 'Rule {\n}\n'
        dsl_str = dsl_str + "Action {{\n  distinctGet({}.__id__ as s, {}.id as s_id, {}.__id__ as o, {}.id as o_id, '{}' as path_type)\n}}\n".format(head, head, tail, tail, path_name)

        return dsl_str

def main():
    path_features = [
        'YAG.YAGOnode_isLocatedIn_YAG.YAGOnode>>>YAG.YAGOnode_isLocatedIn_YAG.YAGOnode>>>YAG.YAGOnode_isLocatedIn_YAG.YAGOnode_inv;o>>>YAG.YAGOnode_hasCapital_YAG.YAGOnode',
    ]

    udf = PathFeatureToDSLUDF()
    for path_feature in path_features:
        print('--------------------------------------------------')
        print(path_feature)
        result = udf.evaluate(path_feature, path_feature, 'tail')
        print('DSL:')
        print(result)

if __name__ == '__main__':
    main()
