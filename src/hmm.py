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
            table[row, col] = count_dict[key1][key2] + 1  # smoothing


def count_starting_state(dict, state):
    if state not in dict:
        dict[state] = 1
    else:
        dict[state] += 1


def init_first_observ_table(table, count_dict, col_lookup):
    for key in count_dict.iterkeys():
        col = col_lookup[key]
        table[col] = count_dict[key]


class HMM(Classifier):

    def __init__(self, new_model=None):
        super(HMM, self).__init__(new_model)
        self.observ_to_id = {}
        self.state_to_id = {}
        self.feature_count_table = np.zeros((1, 1))
        self.transition_count_table = np.zeros((1, 1))
        self.first_observ_count_table = np.zeros(1)
        self.emission_matrix = np.zeros((1, 1))
        self.transition_matrix = np.zeros((1, 1))
        self.first_observ_matrix = np.zeros(1)


    def _collect_counts(self, instance_list):
        """
            The function initislizes 3 tables:
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
            trans_count for transition_count_table, that maps: StateA -> {StateB->count}
            (the meaning is: # times we saw a transition: StateA to StateB)

            observ_count for feature_count_table, that maps: Observation -> {State->count}
            (the meaning is: # times we saw an emission: State -> Observation
            note: the ordering is inversed for a purpose. It helps to deal with both
            dictionaries in the same way in init_table function)

            Smoothing is done by adding 1 to each cell in the matrices.

            3. self.first_observ_count_table -  dimension: 1 X num_states
               C(i) = number of times State_i abbears in the beginning of the sequence

                    |  STATE_1  |  STATE_2  |  ... |  STATE_N  |
                    |-----------|-----------|------|-----------|
                    |   C(1)    |    C(2)   |  ... |    C(N)   |
                    |-----------|-----------|------|-----------|

            This table is used to determine the probabilities of starting the
            hidden state sequence at a particular state.

            Smoothng: transition and feature count tables are calculated in the way that for each count:
                      Count = Real_Count + 1
                      The first_observ_count_table is not being smoothed, because we want to be able to
                      remain with possibilities that equal to 0 for certain states.
        """
        trans_count = {}
        observ_count = {}
        first_state_count = {}

        for instance in instance_list:
            prev_state = None
            #TODO sparse vector like multinomial or binomial?
            sparse_vec = set()
            features = instance.features()
            for i in xrange(len(features)):
                cur_state = instance.label[i]
                update_map(self.state_to_id, cur_state)
                if prev_state is None:
                    count_starting_state(first_state_count, cur_state)
                else:
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
        # init transition and feature count tables with ones for smoothing purpose
        self.transition_count_table = np.ones((num_states, num_states))
        self.feature_count_table = np.ones((num_observ, num_states))
        self.first_observ_count_table = np.zeros(num_states)

        init_table(self.transition_count_table, trans_count, self.state_to_id, self.state_to_id)
        init_table(self.feature_count_table, observ_count, self.observ_to_id, self.state_to_id)
        init_first_observ_table(self.first_observ_count_table, first_state_count, self.state_to_id)

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
        self.init_probability_matrices()

        self.compute_observation_loglikelihood(instance_list[0])


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
            Executes either Forward or Viterbi algorithms.
            1. Forward algorithm
               The trellis is a matrix (Observed_seq_len X num_states)
               Alpha_t(Sj) = P(O1,O2,...,Ot, q_t=State_j)

                     |  STATE_1  |  STATE_2  |  ... |  STATE_N  |
                     |-----------|-----------|------|-----------|
            OBSERV_0 | Alpha0(S1)| Alpha0(S2)|  ... |Alpha0(SN) |
            OBSERV_1 | Alpha1(S1)| Alpha1(S2)|  ... |Alpha1(SN) |
               .     |           |           |  ... |           |
            OBSERV_M | AlphaM(S1)| AlphaM(S2)|  ... | AlphaM(SN)|
                     |-----------|-----------|------|-----------|

            The first row is initialized using the first_observ_matrix probabilities.
            Each following row is computed by:
            Row_k = (Row_(k-1) * transition_matrix) .* relevant_emission_probobilities

            For example, for states B, I, O, computing Row_k = (B_k, I_k, O_k)

                                           (B->B, B->I, B->O)
            temp = (B_k-1, I_k-1, O_k-1) * (I->B, I->I, I->O) = (B'_k, I'_k, O'_k)
                                           (O->B, O->I, O->O)

            Row_k = temp .* (B->Observ_k, I->Observ_k, O->Observ_k)


            Returns trellis filled up with the forward probabilities
            and backtrace pointers for finding the best sequence.
        """
        observ_seq_len = len(instance.features())
        num_states = len(self.state_to_id)
        trellis = np.zeros((observ_seq_len, num_states))
        backtrace_pointers = np.zeros((observ_seq_len, num_states))

        if run_forward_alg:
            # base case initialization for the first row ot the trellis.
            # For each possible State: Alpha_0(State) = P(State)*P(Obs_0 | State)
            first_feature_row = self.observ_to_id[instance.features()[0]]
            trellis[0, :] = self.first_observ_matrix * self.emission_matrix[first_feature_row, :]

            # iteratively filling the rows of the trellis
            for i in range(1, observ_seq_len):
                row = self.observ_to_id[instance.features()[i]]
                temp = np.dot(trellis[i-1, :], self.transition_matrix)
                trellis[i, :] = temp * self.emission_matrix[row, :]

        else:  # run Viterbi algorithm
            pass

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

    def init_probability_matrices(self):
        self.transition_matrix = self.transition_count_table / self.transition_count_table.sum(axis=1, keepdims=True)
        self.emission_matrix = self.feature_count_table / self.feature_count_table.sum(axis=0, keepdims=True)
        self.first_observ_matrix = self.first_observ_count_table / self.first_observ_count_table.sum()

