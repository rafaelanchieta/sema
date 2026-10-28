# -*- coding: utf-8 -*-
"""
This script calculates SEMA score between two AMRs
"""
import argparse
import codecs
import logging
from collections import Counter, defaultdict

from amr import AMR


class Sema:
    """
    SEMA: an Extended Semantic Metric Evaluation for AMR.
    """
    def __init__(self):
        """
        Constructor. Initiates the variables
        """
        # general values
        self.m, self.t, self.c = 0.0, 0.0, 0.0

        # top values
        self.top_m, self.top_t, self.top_c = 0.0, 0.0, 0.0

        # concept values
        self.concept_m, self.concept_t, self.concept_c = 0.0, 0.0, 0.0

        # relation values
        self.relation_m, self.relation_t, self.relation_c = 0.0, 0.0, 0.0

    def compute_sema(self, amr_test, amr_gold):
        """
        Calculates sema metric. Evaluates the test AMR against reference AMR
        :param amr_test: test AMR
        :param amr_gold: reference AMR
        :return:
        """
        test_concepts, test_attributes, test_relations = amr_test.get_triples()
        gold_concepts, gold_attributes, gold_relations = amr_gold.get_triples()

        # remove TOP relation
        test_attributes, gold_attributes = self.remove_top_relation(test_attributes, gold_attributes)

        node_alignment = self._align_nodes(
            test_concepts, test_relations, test_attributes,
            gold_concepts, gold_relations, gold_attributes
        )
        self.m += self._count_matching_triples(
            test_concepts, test_relations, test_attributes,
            gold_concepts, gold_relations, gold_attributes,
            node_alignment
        )
        self.c += self._get_total_triples(test_concepts, test_relations, test_attributes)
        self.t += self._get_total_triples(gold_concepts, gold_relations, gold_attributes)
        print(f'Correct: {self.m} Total: {self.t} Produced: {self.c}')
        print(f'Total test triples: {test_attributes}')

    @staticmethod
    def remove_duplicate_relation(test_relations, gold_relations):
        """
        Removes duplicate relations (Not used)
        :param test_relations: test relations
        :param gold_relations: reference relations
        :return: relations without duplicates
        """
        return list(set(test_relations)), list(set(gold_relations))

    @staticmethod
    def _get_total_triples(concepts, relations, attributes):
        """
        Gets the total number of triples
        :param concepts: concepts
        :param relations: relations
        :param attributes: attributes
        :return: total number of triples
        """
        return len(concepts) + len(relations) + len(attributes)

    @staticmethod
    def _get_previous_node(pre_g, gold_concepts):
        """
        Gets the previous node
        :param pre_g: previous gold node
        :param gold_concepts: gold concepts
        :return: gold node
        """
        node_g = [(rel, var, nod) for rel, var, nod in gold_concepts if var == pre_g][0]
        return node_g

    @staticmethod
    def _check_dependence(pre_t, pre_g, test_concepts, gold_concepts):
        """
        Checks if the nodes of the test and reference AMRs are dependents
        :param pre_t: previous test node
        :param pre_g: previous reference node
        :param test_concepts: test AMR concepts
        :param gold_concepts: reference AMR concepts
        :return: true if nodes are equal
        """
        node_t = [(rel, var, nod) for rel, var, nod in test_concepts if var == pre_t][0][2]
        node_g = [(rel, var, nod) for rel, var, nod in gold_concepts if var == pre_g][0][2]
        if node_t == node_g:
            return True
        return False

    @staticmethod
    def _get_neighbors(pos_t=None, pos_g=None, test_concepts=None, gold_concepts=None):
        """
        Gets neighbors from node
        :param pos_t: posterior test node
        :param pos_g: posterior reference node
        :param test_concepts: test concepts
        :param gold_concepts: reference concepts
        :return: All neighbors nodes
        """
        test_concept = [(rel, var, nod) for rel, var, nod in test_concepts if var == pos_t]
        gold_concept = [(rel, var, nod) for rel, var, nod in gold_concepts if var == pos_g]
        return [test_concept[0][2]], test_concept[0], [gold_concept[0][2]]

    @staticmethod
    def remove_top_relation(test_attributes, gold_attributes):
        """
        Removes :TOP relation from test and reference AMRs
        :param test_attributes: test attributes
        :param gold_attributes: reference attributes
        :return: attributes without :TOP relation
        """
        index1 = [y[0] for y in test_attributes].index('TOP')
        index2 = [y[0] for y in gold_attributes].index('TOP')
        del test_attributes[index1]
        del gold_attributes[index2]
        return test_attributes, gold_attributes

    @staticmethod
    def _maximum_weight_assignment(weights):
        """Return the best column assignment for each row (rows <= columns)."""
        row_count = len(weights)
        if row_count == 0:
            return []
        column_count = len(weights[0])
        max_weight = max(max(row) for row in weights)
        costs = [[max_weight - weight for weight in row] for row in weights]
        row_potential = [0] * (row_count + 1)
        column_potential = [0] * (column_count + 1)
        matched_row = [0] * (column_count + 1)
        previous_column = [0] * (column_count + 1)

        for row in range(1, row_count + 1):
            matched_row[0] = row
            current_column = 0
            min_cost = [float('inf')] * (column_count + 1)
            used = [False] * (column_count + 1)
            while True:
                used[current_column] = True
                current_row = matched_row[current_column]
                delta = float('inf')
                next_column = 0
                for column in range(1, column_count + 1):
                    if used[column]:
                        continue
                    cost = (
                        costs[current_row - 1][column - 1]
                        - row_potential[current_row]
                        - column_potential[column]
                    )
                    if cost < min_cost[column]:
                        min_cost[column] = cost
                        previous_column[column] = current_column
                    if min_cost[column] < delta:
                        delta = min_cost[column]
                        next_column = column
                for column in range(column_count + 1):
                    if used[column]:
                        row_potential[matched_row[column]] += delta
                        column_potential[column] -= delta
                    else:
                        min_cost[column] -= delta
                current_column = next_column
                if matched_row[current_column] == 0:
                    break
            while True:
                next_column = previous_column[current_column]
                matched_row[current_column] = matched_row[next_column]
                current_column = next_column
                if current_column == 0:
                    break

        assignment = [0] * row_count
        for column in range(1, column_count + 1):
            if matched_row[column] != 0:
                assignment[matched_row[column] - 1] = column - 1
        return assignment

    @classmethod
    def _align_nodes(cls, test_concepts, test_relations, test_attributes,
                     gold_concepts, gold_relations, gold_attributes):
        """Build a one-to-one node mapping, preferring structurally similar nodes."""
        test_values = {var: value for _, var, value in test_concepts}
        gold_values = {var: value for _, var, value in gold_concepts}

        def node_features(concepts, relations, attributes):
            values = {var: value for _, var, value in concepts}
            attributes_by_node = defaultdict(Counter)
            outgoing = defaultdict(Counter)
            incoming = defaultdict(Counter)
            outgoing_context = defaultdict(Counter)
            incoming_context = defaultdict(Counter)
            for relation, node, value in attributes:
                attributes_by_node[node][(relation, value)] += 1
            for relation, parent, child in relations:
                outgoing[parent][(relation, values[child])] += 1
                incoming[child][(relation, values[parent])] += 1
                outgoing_context[parent][
                    (relation, values[child], tuple(sorted(attributes_by_node[child].items())))
                ] += 1
                incoming_context[child][
                    (relation, values[parent], tuple(sorted(attributes_by_node[parent].items())))
                ] += 1
            return attributes_by_node, outgoing, incoming, outgoing_context, incoming_context

        test_features = node_features(test_concepts, test_relations, test_attributes)
        gold_features = node_features(gold_concepts, gold_relations, gold_attributes)

        def similarity(test_node, gold_node):
            score = 1
            for feature_index, weight in ((0, 5), (1, 2), (2, 1), (3, 4), (4, 2)):
                test_feature = test_features[feature_index][test_node]
                gold_feature = gold_features[feature_index][gold_node]
                score += weight * sum((test_feature & gold_feature).values())
            return score

        alignment = {}
        test_root = test_concepts[0][1]
        gold_root = gold_concepts[0][1]
        if test_values[test_root] == gold_values[gold_root]:
            alignment[test_root] = gold_root

        test_nodes_by_concept = defaultdict(list)
        gold_nodes_by_concept = defaultdict(list)
        for _, node, concept in test_concepts:
            if node not in alignment:
                test_nodes_by_concept[concept].append(node)
        for _, node, concept in gold_concepts:
            if node not in alignment.values():
                gold_nodes_by_concept[concept].append(node)

        for concept, test_nodes in test_nodes_by_concept.items():
            gold_nodes = gold_nodes_by_concept.get(concept, [])
            if not gold_nodes:
                continue
            weights = [
                [similarity(test_node, gold_node) for gold_node in gold_nodes]
                for test_node in test_nodes
            ]
            if len(test_nodes) <= len(gold_nodes):
                assignment = cls._maximum_weight_assignment(weights)
                alignment.update({
                    test_nodes[row]: gold_nodes[column]
                    for row, column in enumerate(assignment)
                })
            else:
                transposed = [
                    [weights[row][column] for row in range(len(test_nodes))]
                    for column in range(len(gold_nodes))
                ]
                assignment = cls._maximum_weight_assignment(transposed)
                alignment.update({
                    test_nodes[column]: gold_nodes[row]
                    for row, column in enumerate(assignment)
                })
        return alignment

    @staticmethod
    def _count_matching_triples(test_concepts, test_relations, test_attributes,
                                gold_concepts, gold_relations, gold_attributes,
                                alignment):
        """Count exact triples after translating test nodes to aligned gold nodes."""
        def count_matches(source, target, translate):
            available = Counter(target)
            count = 0
            for triple in source:
                translated = translate(triple)
                if translated is not None and available[translated]:
                    available[translated] -= 1
                    count += 1
            return count

        matched = count_matches(
            test_concepts, gold_concepts,
            lambda triple: (triple[0], alignment[triple[1]], triple[2])
            if triple[1] in alignment else None
        )
        matched += count_matches(
            test_relations, gold_relations,
            lambda triple: (triple[0], alignment[triple[1]], alignment[triple[2]])
            if triple[1] in alignment and triple[2] in alignment else None
        )
        matched += count_matches(
            test_attributes, gold_attributes,
            lambda triple: (triple[0], alignment[triple[1]], triple[2])
            if triple[1] in alignment else None
        )
        return matched

    def compute_root(self, test_root, gold_root, visited):
        """
        Computes the root node
        :param test_root: test root
        :param gold_root: reference root
        :param visited: visited triples
        :return:
        """
        if test_root == gold_root:
            self.m += 1
            self.top_m += 1
            visited.append(('instance', 'b0', gold_root))

    def compute_relations(self, test_relations, gold_relations, test_concepts, gold_concepts, visited):
        """
        Computes relation triples take into account its dependence
        :param test_relations: test relations
        :param gold_relations: reference relations
        :param test_concepts: test concepts
        :param gold_concepts: reference concepts
        :param visited: visited triples
        :return:
        """
        for rel_t, pre_t, pos_t in test_relations:
            for rel_g, pre_g, pos_g in gold_relations:
                # if relations are equal
                if rel_t == rel_g:
                    # if previous concepts are equal
                    if self._check_dependence(pre_t, pre_g, test_concepts, gold_concepts):
                        # gets neighbors
                        test_concept, current_node, gold_concept = self._get_neighbors(pos_t=pos_t, pos_g=pos_g,
                                                                                       test_concepts=test_concepts,
                                                                                       gold_concepts=gold_concepts)
                        relation = (rel_t, pre_t, pos_t)  # (rel_g, pre_g, pos_g)
                        previous_node = self._get_previous_node(pre_t, test_concepts)  # (pre_g, gold_concepts)
                        self.compute_concepts(test_concept, gold_concept, current_node, relation, previous_node,
                                              visited)

    def compute_concepts(self, test_concepts, gold_concepts, current_node, relation, previous_node, visited):
        """
        Computes the concepts take into account its dependence
        :param test_concepts: test concepts
        :param gold_concepts: reference concepts
        :param current_node: current node
        :param relation: previous relation
        :param previous_node: previous node
        :param visited: visited triples
        :return:
        """
        for test_node in test_concepts:
            for gold_node in gold_concepts:
                if test_node == gold_node:
                    if current_node not in visited and relation not in visited:
                        self.m += 2
                        visited.append(current_node)
                        visited.append(relation)
                    if previous_node not in visited:
                        self.m += 1
                        visited.append(previous_node)
                    if current_node in visited and previous_node in visited and relation not in visited:
                        flag = False
                        for triple in visited:
                            if relation[1] == triple[1] and relation[2] == triple[2]:
                                flag = True
                        if not flag:
                            self.m += 1
                            visited.append(relation)

    def compute_attributes(self, test_attributes, gold_attributes, test_concepts, gold_concepts, visited):
        """
        Computes attributes test triples against attributes reference triples
        :param test_attributes: test attributes
        :param gold_attributes: reference attributes
        :param test_concepts: test concepts
        :param gold_concepts: reference concepts
        :param visited: visited triples
        :return:
        """
        for rel_t, pre_t, pos_t in test_attributes:
            for rel_g, pre_g, pos_g in gold_attributes:
                # if relation and constant are equal to reference values
                if self._check_dependence(pre_t, pre_g, test_concepts, gold_concepts):
                    if rel_t == rel_g and pos_t == pos_g and (rel_g, pre_g, pos_g) not in visited:
                        self.m += 1
                        visited.append((rel_g, pre_g, pos_g))

    def get_sema_value(self):
        """
        Calculates precision, recall, and f-score
        precision   = correct triples / produced triples
        recall      = correct triples / total triples
        :return: precision, recall, and f-score
        """
        try:
            precision = self.m / self.c
            recall = self.m / self.t
            f1 = 2 * precision * recall / (precision + recall)
            return precision, recall, f1
        except ZeroDivisionError:
            return 0, 0, 0


def main(data):
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger(__name__)
    test = codecs.open(data.test, 'r', 'utf-8')
    gold = codecs.open(data.gold, 'r', 'utf-8')
    flag = False
    sema = Sema()
    while True:
        cur_amr1 = AMR.get_amr_line(test)
        cur_amr2 = AMR.get_amr_line(gold)

        if cur_amr1 == '' and cur_amr2 == '':
            break
        if cur_amr1 == '':
            logger.error('Error: File 1 has less AMRs than file 2')
            logger.error('Ignoring remaining AMRs')
            flag = True
            break
        if cur_amr2 == '':
            logger.error('Error: File 2 has less AMRs than file 1')
            logger.error('Ignoring remaining AMRs')
            flag = True
            break
        try:
            amr1 = AMR.parse_AMR_line(cur_amr1)
        except Exception as e:
            logger.error('Error in parsing amr 1: %s' % cur_amr1)
            logger.error("Please check if the AMR is ill-formatted. Ignoring remaining AMRs")
            logger.error("Error message: %s" % str(e))
            flag = True
            break
        try:
            amr2 = AMR.parse_AMR_line(cur_amr2)
        except Exception as e:
            logger.error("Error in parsing amr 2: %s" % cur_amr2)
            logger.error("Please check if the AMR is ill-formatted. Ignoring remaining AMRs")
            logger.error("Error message: %s" % str(e))
            flag = True
            break
        prefix_test = 'a'
        prefix_gold = 'b'
        amr1.rename_node(prefix_test)
        amr2.rename_node(prefix_gold)
        sema.compute_sema(amr1, amr2)
    if not flag:
        precision, recall, f1 = sema.get_sema_value()
        print(f'SEMA: P {precision:.2f} R {recall:.2f} F1 {f1:.2f}')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='SEMA metric',
                                   epilog='Usage: python sema.py -t parsedAMR.txt -g referenceAMR.txt')
    args.add_argument('-t', '--test', help='test AMR', required=True)
    args.add_argument('-g', '--gold', help='Reference AMR', required=True)
    main(args.parse_args())
