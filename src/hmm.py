from __future__ import division
import numpy as np

from classifier import Classifier
# from sets import Set

def get_sparse_vector(fnames, map_name_to_id):
    names_set = set(fnames)
    sparse_vector = []
    for name in names_set:
        if name in map_name_to_id:
            sparse_vector.append(map_name_to_id[name])
    return sparse_vector


def add_count(dict, from_val, to_val):
    if from_val is None or to_val is None: return
    if from_val not in dict:
        dict[from_val] = {to_val: 1}
    elif to_val not in dict[from_val]:
        dict[from_val][to_val] = 1
    else:
        dict[from_val][to_val] += 1


def update_map(map, entry):
    if entry not in map:
        map[entry] = len(map)


def init_table(table, count_dict, row_lookup, col_lookup):
    for key1 in count_dict.iterkeys():
        row = row_lookup[key1]
        for key2 in count_dict[key1].iterkeys():
            col = col_lookup[key2]
            table[row, col] = count_dict[key1][key2]


class HMM(Classifier):

    def __init__(self, new_model=None):
        super(HMM, self).__init__(new_model)
        self.observ_to_id = {}
        self.state_to_id = {}
        self.feature_count_table = np.zeros((1, 1))
        self.transition_count_table = np.zeros((1, 1))
        self.emission_matrix = np.zeros((1, 1))
        self.transition_matrix = np.zeros((1, 1))


    def _collect_counts(self, instance_list):
        """
            The function initislizes 2 tables:
            1. self.transition_count_table -  dimension: num_states X num_states
               A(i,j) = number of i->j transitions for sample DB

                    |  STATE_1  |  STATE_2  |  ... |  STATE_N  |
                    |-----------|-----------|------|-----------|
            STATE_1 |  A(1,1)   |  A(1,2)   |  ... |  A(1,N)   |
            STATE_2 |  A(2,1)   |  A(2,2)   |  ... |  A(2,N)   |
               .    |           |           |  ... |           |
               .    |           |           |  ... |           |
            STATE_N |  A(N,1)   |  A(N,2)   |  ... |  A(N,N)   |
                    |-----------|-----------|------|-----------|

            2. self.feature_count_table -  dimension: num_features X num_states
                E(S,i) = number of emissions of observation S at state i

                     |  STATE_1  |  STATE_2  |  ... |  STATE_N  |
                     |-----------|-----------|------|-----------|
            OBSERV_1 |  E(1,1)   |  E(1,2)   |  ... |  E(1,N)   |
            OBSERV_2 |  E(2,1)   |  E(2,2)   |  ... |  E(2,N)   |
               .     |           |           |  ... |           |
               .     |           |           |  ... |           |
               .     |           |           |  ... |           |
               .     |           |           |  ... |           |
               .     |           |           |  ... |           |
            OBSERV_M |  E(M,1)   |  E(M,2)   |  ... |  E(M,N)   |
                     |-----------|-----------|------|-----------|

            This is done by creating 2 nested dictionaries.
            One for transition_count_table, that maps: StateA -> {StateB->count}
            (the meaning is: # times we saw transition StateA to StateB)
            Another for feature_count_table, that maps: Observation -> {State->count}
            (the meaning is: # times we saw emission State -> Observation)



        """
        trans_count = {}
        observ_count = {}

        for instance in instance_list:
            prev_state = None
            #TODO sparse vector like multinomial or binomial?
            sparse_vec = set()
            features = instance.features()
            for i in xrange(len(features)):
                cur_state = instance.label[i]
                update_map(self.state_to_id, cur_state)
                add_count(trans_count, prev_state, cur_state)
                prev_state = cur_state

                current_observ = features[i]
                update_map(self.observ_to_id, current_observ)
                sparse_vec.add(self.observ_to_id[current_observ])
                add_count(observ_count, current_observ, cur_state)
            # cache sparse feature vector for each instance
            instance.feature_vector = list(sparse_vec)

        num_states = len(self.state_to_id)
        num_observ = len(self.observ_to_id)
        self.transition_count_table = np.zeros((num_states, num_states))
        self.feature_count_table = np.zeros((num_observ, num_states))

        init_table(self.transition_count_table, trans_count, self.state_to_id, self.state_to_id)
        init_table(self.feature_count_table, observ_count, self.observ_to_id, self.state_to_id)


    def train(self, instance_list, ):
        """
            Fit parameters for hidden markov model

            Update codebooks from the given data to be consistent with
            the probability tables

            Transition matrix and emission probability matrix
            will then be populated with the maximum likelihood estimate
            of the appropriate parameters

            Add your docstring here explaining how you implement this function

            Returns None
        """
        self._collect_counts(instance_list)

        # TODO: estimate the parameters from the count tables

    def classify(self, instance):
        """
            Viterbi decoding algorithm

            Wrapper for running the Viterbi algorithm
            We can then obtain the best sequence of labels from the backtrace pointers matrix

            Add your docstring here explaining how you implement this function

            Returns a list of labels e.g. ['B','I','O','O','B']
        """
        backtrace_pointers = self.dynamic_programming_on_trellis(instance, False)
        best_sequence = []
        return best_sequence

    def compute_observation_loglikelihood(self, instance):
        """
            Compute and return log P(X|parameters) = loglikelihood of observations
        """
        trellis = self.dynamic_programming_on_trellis(instance, True)
        loglikelihood = 0.0
        return loglikelihood

    def dynamic_programming_on_trellis(self, instance, run_forward_alg=True):
        """
            Run Forward algorithm or Viterbi algorithm
            This function uses the trellis to implement dynamic
            programming algorithm for obtaining the best sequence
            of labels given the observations

            Add your docstring here explaining how you implement this function

            Returns trellis filled up with the forward probabilities
            and backtrace pointers for finding the best sequence
        """
        # TODO:Initialize trellis and backtrace pointers
        trellis = np.zeros((1, 1))
        backtrace_pointers = np.zeros((1, 1))
        # TODO:Traverse through the trellis here

        return trellis, backtrace_pointers

    def train_semisupervised(self, unlabeled_instance_list, labeled_instance_list=None):
        """
            Baum-Welch algorithm for fitting HMM from unlabeled data (EXTRA CREDIT)

            The algorithm first initializes the model with the labeled data if given.
            The model is initialized randomly otherwise. Then it runs
            Baum-Welch algorithm to enhance the model with more data.

            Add your docstring here explaining how you implement this function

            Returns None
        """
        if labeled_instance_list is not None:
            self.train(labeled_instance_list)
        else:
            # TODO: initialize the model randomly
            pass
        while True:
            # E-Step
            self.expected_transition_counts = np.zeros((1, 1))
            self.expected_feature_counts = np.zeros((1, 1))
            for instance in instance_list:
                (alpha_table, beta_table) = self._run_forward_backward(instance)
            # TODO: update the expected count tables based on alphas and betas
            # also combine the expected count with the observed counts from the labeled data
            # M-Step
            # TODO: reestimate the parameters
            if self._has_converged(old_likelihood, likelihood):
                break

    def _has_converged(self, old_likelihood, likelihood):
        """
            Determine whether the parameters have converged or not (EXTRA CREDIT)

            Returns True if the parameters have converged.
        """
        return True

    def _run_forward_backward(self, instance):
        """
            Forward-backward algorithm for HMM using trellis (EXTRA CREDIT)

            Fill up the alpha and beta trellises (the same notation as
            presented in the lecture and Martin and Jurafsky)
            You can reuse your forward algorithm here

            return a tuple of tables consisting of alpha and beta tables
        """
        alpha_table = np.zeros((1, 1))
        beta_table = np.zeros((1, 1))
        # TODO: implement forward backward algorithm right here

        return (alpha_table, beta_table)

    def get_model(self):
        return None

    def set_model(self, model):
        pass

    model = property(get_model, set_model)

