'''
Program: Security Trading Using Neural Networks
Author: Connor S. Maynes
Purpose: Generate neural networks from security data and predict future prices.
'''

import pandas.io.data as web  # Package and modules for importing data; this code may change depending on pandas version
import pandas as pd
from datetime import date, datetime, timedelta
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
from cmd2 import Cmd
from tqdm import tqdm, trange
from time import sleep
import os


class Network_Builder(Cmd):
    '''Build/work with the Network class.'''

    def __init__(self):

        Cmd.__init__(self, use_ipython=True)
        self.network = Network()

    @classmethod
    def do_load_network(self):
        '''Load network from file.'''

        network_location = r'' + input("Please type the address of the network's file: ")
        self.network = pickle.load(open(network_location, 'rb'))

        return True

    @classmethod
    def do_save_network(self):
        '''Save the Network object using pickle library.'''

        if 'y' in raw_input('Default save settings?(y/n) '):
            file_name = self.network.name
            save_location = self.network.save_path

        else:
            file_name = raw_input('Name file: ')
            save_location = raw_input('Type file save location: ')

        save_path = save_location + '/' + file_name + '.p'
        print('Saving network...')
        pickle.dump(self.network, open(save_path, 'wb'))
        print('Network successfully saved to: ' + save_path)
        return True


class Network:
    '''The Neural Network houses all of the Receptor, Neuron, and Effector objects.'''

    SECURITY_START_DATE = datetime(year=1900, month=1, day=1)
    SECURITY_END_DATE = datetime(date.today().year,date.today().month,date.today().day)

    def __init__(self):

        self.input_vectors = None
        self.output_vectors = None
        self.vector_date_list = None
        self.input_vectors_training_set = None
        self.input_vectors_testing_set = None
        self.output_vectors_training_set = None
        self.output_vectors_testing_set = None
        self.vector_date_list_training_set = None
        self.vector_date_list_testing_set = None
        self.neuron_matrix_rows = None
        self.neuron_matrix_columns = None
        self.optimized = False
        self.trained_on_function = False
        self.trained_on_security = False
        self.graphed_outputs = False
        self.save_path = 'C:/'
        self.short_info = None
        self.optimizer_median_errors_dict = {}
        self.info = None

        self.weights = []  # The weights for each layer
        self.outputs = []
        self.inputs = []
        self.delta_errors = []
        self.vector_output_error = []

    def do_build_new(self):
        '''Build a new Network. Set the parameters. Guided setup.'''

        train_on_function = 'n'
        train_on_security = 'y'
        new_or_load_network = raw_input('New (n) or load (l) network? ')

        if 'n' in new_or_load_network.lower():

            self.save_path = r'' + raw_input('Save location? ')

            train_on_function = raw_input('Train on function?(y/n) ').lower()
            if 'y' in train_on_function:
                self.trained_on_function = True
                optimize = raw_input('Optimize for function?(y/n) ').lower()
                if 'y' in optimize:
                    self.optimized = True

            else:
                self.trained_on_security = True
                optimize = raw_input('Optimize for security?(y/n) ').lower()
                if 'y' in optimize:
                    self.optimized = True

            use_defaults = raw_input('Use default settings?(y/n) ').lower()
            good_settings = 'n'
            while 'n' in good_settings:    # allow user opportunity to change settings if they make a mistake

                if 'y' in use_defaults:
                    self.do_set_defaults() # use predetermined default values

                else:   # if not using defaults
                    if self.trained_on_function:
                        self.do_set_network_parameters(security='SINE',
                                                    input_vector_parameters=['x'],
                                                    output_vector_parameters=['SINE'])
                    else:
                        self.do_set_network_parameters()

                print('')
                print('*** Here are the network settings you have selected ***')
                for i in self.do_get_info():
                    print(i)
                print('*** End network parameters ***')
                print('')

                good_settings = raw_input('Good settings?(y/n) ')
                if 'n' in good_settings:    # allow user opportunity to make adjustments to settings
                    use_defaults = 'n'

            graph_outputs = raw_input('Graph outputs?(y/n) ').lower()
            if 'y' in graph_outputs:
                self.graphed_outputs = True

        else:   # load the Network from a file using Pickle

            Network_Builder.do_load_network() # request the Network file and load using pickle
            print('Network loaded from file!')

        print('')
        print('Initializing network...')
        print('')

        self.prediction_log = {'epoch':[],'learned_from':[],'security':[],'prediction_start_date':[],'input_vector':[],
                               'expected_output_vector':[],'prediction_end_dates':[],'actual_output_vectors':[],
                               'errors':[]}  # store all of the predictions the network ever makes

        print('epoch', 'learned_from', 'security', 'prediction_start_date', 'input_vector', 'expected_output_vector',
         'prediction_end_dates', 'actual_output_vector', 'errors')

        if self.optimized:
            self.do_optimize()

        elif train_on_function:

            self.build()    # build the empty network. Initialize weights with random values.
            self.do_train_on_function()

        elif train_on_security:

            self.build()
            self.do_train_on_security()

    def do_delete_network(self):
        '''Delete the network matrices as well as some other settings and information.'''

        self.do_delete_network_matrices()

        self.name = None
        self.security = None
        self.output_vector_parameters = None
        self.input_vector_parameters = None
        self.neuron_activation_function = None
        self.effector_activation_function = None
        self.training_testing_ratio = None
        self.learning_rate = None
        self.network_dimensions = None

    def do_delete_network_matrices(self):
        '''Delete the Network matrices '''

        self.weights = []

        self.inputs = []
        self.input_vectors = []
        self.input_vectors_training_set = []
        self.input_vectors_testing_set = []

        self.outputs = []
        self.output_vectors = []
        self.output_vectors_training_set = []
        self.output_vectors_testing_set = []

        self.delta_errors = []

        self.vector_date_list = []
        self.vector_date_list_training_set = []
        self.vector_date_list_testing_set = []

        return True

    def do_set_network_parameters(self,
                               name=None,
                               security=None,
                               training_testing_ratio=None,
                               learning_rate=None,
                               input_vector_parameters=None,
                               neuron_matrix_rows=None,
                               neuron_matrix_columns=None,
                               output_vector_parameters=None,
                               neuron_activation_function=None,
                               effector_activation_function=None):
        '''Delete existing Network settings. Ask for arguments from the user for the Network. Return True is successful.
        '''

        SECURITY_INPUT_VECTOR_PARAMETERS = ['Close', 'Open', 'Volume', 'Adj Close', 'ndma -- for some number of days n']

        self.do_delete_network_matrices()

        if not name:
            self.name = raw_input('Name your network: ')
        else:
            self.name = name

        if not security:
            good_ticker = False
            security_input = None
            while not good_ticker:  # verify the ticker symbol exists before continuing
                security_input = raw_input('Ticker of security to train network on: ').upper()
                try:
                    temp_data = web.DataReader(security_input, 'yahoo', '2010/01/01', date.today())['Close']
                    good_ticker = True
                except:
                    print('Error: could not find ticker.')
            self.security = security_input
        else:
            self.security = security

        if not input_vector_parameters:
            print('')
            print('Input Vector Parameter Options: ')
            print('')
            for input_vector in SECURITY_INPUT_VECTOR_PARAMETERS:
                print('    ' + input_vector)
            self.input_vector_parameters = raw_input('List the input vector parameters(space-separate): ').split()
            print('')
        else:
            self.input_vector_parameters = input_vector_parameters
        receptor_matrix_rows = int(np.size([0]))
        receptor_matrix_columns = int(len(self.input_vector_parameters))

        if not neuron_matrix_rows:
            z = ''
            while not z.isdigit():
                z = raw_input('Number of rows of Neurons: ')
            neuron_matrix_rows = int(z)

        if not neuron_matrix_columns:
            z = ''
            while not z.isdigit():
                z = raw_input('Number of columns of Neurons: ')
            neuron_matrix_columns = int(z)

        if not training_testing_ratio:
            z = ''
            while not isfloat(z):
                z = raw_input('Training / Testing ratio (decimal): ')
            self.training_testing_ratio = float(z)
        else:
            self.training_testing_ratio = training_testing_ratio

        if not learning_rate:
            z = ''
            while not isfloat(z):
                z = raw_input('Learning rate: ')
            self.learning_rate = float(z)
        else:
            self.learning_rate = learning_rate

        if not output_vector_parameters:
            print('')
            all_ints = False
            while not all_ints:
                output_vector_parameters = \
                    raw_input('Number of datapoints to lookforward (starting data point after prediction; space-separate): ')
                for output_vector in output_vector_parameters.split():
                    if not output_vector.isdigit():
                        all_ints = False
                        break
                    else:
                        all_ints = True
            self.output_vector_parameters = output_vector_parameters
            print('')
        else:
            self.output_vector_parameters = output_vector_parameters
        effector_matrix_rows = int(len(self.output_vector_parameters))
        effector_matrix_columns = int(np.size([0]))

        if not neuron_activation_function:
            print('Activation Function Options: sigmoid, tanh, or linear')
            z = ''
            while not getattr(sys.modules[__name__], z):
                z = raw_input('NEURON activation function? ')
            self.neuron_activation_function = getattr(sys.modules[__name__], z)
        else:
            self.neuron_activation_function = neuron_activation_function

        if not effector_activation_function:
            z = ''
            while not getattr(sys.modules[__name__], z):
                z = raw_input('EFFECTOR activation function? ')
            self.effector_activation_function = getattr(sys.modules[__name__], z)
        else:
            self.effector_activation_function = effector_activation_function

        self.network_dimensions = [[receptor_matrix_rows, neuron_matrix_columns],
                                   [neuron_matrix_rows, neuron_matrix_columns],
                                   [effector_matrix_rows, effector_matrix_columns]]

        self.short_info = self.name + '#nrows-ncols-ratio-rate##' + \
                            str(self.network_dimensions[1][0]) + ' ' + \
                                str(self.network_dimensions[1][1]) + ' ' + \
                                    str(self.training_testing_ratio) + ' ' + \
                                        str(self.learning_rate)

        return True

    def do_set_defaults(self):
        '''Set the Network arguments to the default values. Erases any existing network.'''

        self.do_delete_network_matrices()

        if self.trained_on_security:
            security = 'YHOO'
            input_vector_parameters = ['Close', '3dma', '9dma']
            output_vector_parameters = [1]

        elif self.trained_on_function:
            security = 'SINE'
            input_vector_parameters = ['x']
            output_vector_parameters = [0]

        name = security.lower() + '_network'
        neuron_matrix_rows = 3
        neuron_matrix_columns = 2
        training_testing_ratio = 0.4
        learning_rate = 0.02
        neuron_activation_function = tanh
        effector_activation_function = linear

        self.do_set_network_parameters(name=name,
                                    security=security,
                                    training_testing_ratio=training_testing_ratio,
                                    learning_rate=learning_rate,
                                    neuron_matrix_rows=neuron_matrix_rows,
                                    neuron_matrix_columns=neuron_matrix_columns,
                                    input_vector_parameters=input_vector_parameters,
                                    output_vector_parameters=output_vector_parameters,
                                    neuron_activation_function=neuron_activation_function,
                                    effector_activation_function=effector_activation_function)

        return True

    def generate_security_output_vectors(self):
        '''Generate all of the output vectors, using the user's inputted output vectors.'''

        print('     generating output vectors...')

        output_vectors_list = []

        all_close_prices_list = \
            web.DataReader(self.security, 'yahoo', self.SECURITY_START_DATE, self.SECURITY_END_DATE)['Close']

        for close_index in range(len(all_close_prices_list)):

            temp = []
            for parameter in self.output_vector_parameters:

                close_price_vector_index = close_index + parameter
                if close_price_vector_index <= len(all_close_prices_list) - 1:
                    temp.append(all_close_prices_list[close_price_vector_index])

            output_vectors_list.append(temp)  # add the input vector to the list of input vectors

        return output_vectors_list

    def generate_security_input_vectors(self):
        '''Get the input vectors for the security provided by the user in the form of a list of lists. Return the
        matrix of vectors, with each input vector as a row.'''

        print('     generating input vectors...')

        input_vector_list = []
        temp = []

        for parameter in self.input_vector_parameters:

            if 'ma' in parameter:
                ma_range = int(parameter.split('d', 1)[0])  # get number of days for moving average
                close_prices = web.DataReader(self.security, 'yahoo', self.SECURITY_START_DATE, self.SECURITY_END_DATE)
                ma = pd.rolling_mean(close_prices['Close'], window=ma_range)
                temp.append(ma)  # tack the moving average onto the input vector

            else:
                temp.append(web.DataReader(self.security, 'yahoo', self.SECURITY_START_DATE, self.SECURITY_END_DATE)[parameter])

        # COMBINE THE ITEMS FROM EACH ROW IN EACH LIST OF INPUT VECTOR DATA
        for row in range(len(temp[0])):
            row_vector = []

            for list_index in range(len(temp)):
                row_vector.append(temp[list_index][row])

            input_vector_list.append(row_vector)  # add the input vector to the list of input vectors

        return input_vector_list

    def generate_security_date_list(self):
        '''Generate the list of dates corresponding to the vector inputs and outputs.'''

        print('     generating vector date list...')

        vector_date_list = web.DataReader(self.security, 'yahoo', self.SECURITY_START_DATE, self.SECURITY_END_DATE)[
            'Close'].index

        return vector_date_list

    def build(self):
        '''The neural network requires the input and output of every neuron to be saved. All of the inputs and outputs
        are saved in separate matrices to improve readability.'''

        # matrix of weights for inputs. Number of rows equals number of input vectors. Number of columns is the number
        # of rows in hidden neuron layer
        self.weights.append(np.random.rand(self.network_dimensions[0][0],self.network_dimensions[1][0]))
        # matrix of weights for hidden layer. Number of rows/cols = number of rows of neurons.
        # Each row is a neuron's weights for the following layer. Last weight array for effectors has different shape.
        for x in range(0,self.network_dimensions[1][1] - 1):
            self.weights.append(np.random.rand(self.network_dimensions[1][0], self.network_dimensions[1][0]))
        self.weights.append(np.random.rand(self.network_dimensions[1][0], self.network_dimensions[2][0]))

        self.reset_i_o_de()

        return True

    def reset_i_o_de(self):
        '''Reset the input, output, and delta error matrices to empty matrices of zeroes.'''

        self.inputs = []
        # empty array for receptor neurons (inputs)
        self.inputs.append(np.zeros((np.size([0]), self.network_dimensions[0][1])))
        # empty array for hidden neurons
        for x in range(0,self.network_dimensions[1][1]):
            self.inputs.append(np.zeros((self.network_dimensions[1][0], np.size([0]))))
        # empty array for effector neurons (outputs)
        self.inputs.append(np.zeros((self.network_dimensions[2][0], np.size([0]))))

        self.outputs = []
        # empty array for receptor neurons (inputs)
        self.outputs.append(np.zeros((np.size([0]), self.network_dimensions[0][1])))
        # empty array for hidden neurons
        for x in range(0,self.network_dimensions[1][1]):
            self.outputs.append(np.zeros((self.network_dimensions[1][0], np.size([0]))))
        # empty array for effector neurons (outputs)
        self.outputs.append(np.zeros((self.network_dimensions[2][0], np.size([0]))))

        self.delta_errors = []
        # empty array for receptor neurons (inputs)
        self.delta_errors.append(np.zeros((self.network_dimensions[0][1], np.size([0]))))
        # empty array for hidden neurons
        for x in range(0,self.network_dimensions[1][1]):
            self.delta_errors.append(np.zeros((self.network_dimensions[1][0], np.size([0]))))
        # empty array for effector neurons (outputs)
        self.delta_errors.append(np.zeros((self.network_dimensions[2][0], np.size([0]))))

    def fetch_training_testing_vectors(self, input_vectors, output_vectors, vector_date_list=None):
        '''Get all of the data for training and for testing, and chop according in the ratio provided by the user.'''

        training_end_index = int(self.training_testing_ratio * (len(input_vectors)-1))

        input_vectors_training_set = [input_vectors[x] for x in range(training_end_index)]
        input_vectors_testing_set = [input_vectors[training_end_index + x] for x in
                                          range(len(input_vectors)-training_end_index-1)]

        output_vectors_training_set = [output_vectors[x] for x in range(training_end_index)]
        output_vectors_testing_set = [output_vectors[training_end_index + x] for x in
                                           range(len(output_vectors)-training_end_index-1)]

        vector_date_list_training_set = [vector_date_list[x] for x in range(training_end_index)]
        vector_date_list_testing_set = [vector_date_list[training_end_index + x] for x in
                                        range(len(vector_date_list)-training_end_index-1)]

        return input_vectors_training_set, input_vectors_testing_set, output_vectors_training_set, \
               output_vectors_testing_set, vector_date_list_training_set, vector_date_list_testing_set

    def do_train_on_function(self):
        '''Sanity check for the network using a function to generate inputs.'''

        self.input_vectors = np.random.random((1000,1))
        self.output_vectors = np.sin(self.input_vectors)
        self.vector_date_list = \
            [datetime(date.today().year,date.today().month,date.today().day) - timedelta(days=1000-x)
             for x in range(1000)]

        self.input_vectors_training_set, self.input_vectors_testing_set, \
        self.output_vectors_training_set, self.output_vectors_testing_set, \
        self.vector_date_list_training_set, self.vector_date_list_testing_set = \
            self.fetch_training_testing_vectors(input_vectors=self.input_vectors, output_vectors=self.output_vectors,
                                                vector_date_list=self.vector_date_list)

        self.train()

        self.do_plot_outputs()

    def do_export_plot(self, plt=None, plt_name=None):
        '''Save the plot to self.save_path.'''

        plot_folder_path = r'' + self.save_path + '/plots/' # make directory if it does not exist yet
        if not os.path.exists(plot_folder_path):
            os.makedirs(plot_folder_path)

        if plt:
            if not plt_name:
                plt_name = self.name + '_outputs_plot'
            plot_path = r'' + self.save_path + '/plots/' + plt_name + '.jpg'
            plt.savefig(plot_path)

        return True

    def do_train_on_security(self):
        '''Train the neural network, ANN or RC, using the split training and testing data.'''

        self.input_vectors = self.generate_security_input_vectors()  # get list of input vectors
        self.output_vectors = self.generate_security_output_vectors()
        self.vector_date_list = self.generate_security_date_list()

        self.input_vectors_training_set, self.input_vectors_testing_set, self.output_vectors_training_set, \
        self.output_vectors_testing_set, self.vector_date_list_training_set, self.vector_date_list_testing_set = \
            self.fetch_training_testing_vectors(input_vectors=self.input_vectors, output_vectors=self.output_vectors,
                                            vector_date_list=self.vector_date_list)

        self.train()
        
        self.do_plot_outputs()

    def train(self):
        '''Train the network on the current input, output, and date vector data saved to self.'''

        for v in trange(len(self.vector_date_list_training_set), desc='training', leave=False):
            self.forward_propagate(self.input_vectors_training_set[v], self.output_vectors_training_set[v])

            self.back_propagate(self.output_vectors_training_set[v])

            self.log_prediction(self.vector_date_list_training_set[v], self.input_vectors_training_set[v],
                                self.output_vectors_training_set[v], learned_from=True)

            self.reset_i_o_de() # reset the input, output and delta error matrices back to matrices of zeroes

            #sleep(0.005)

        for v in trange(len(self.vector_date_list_testing_set), desc='testing', leave=False):
            self.forward_propagate(self.input_vectors_testing_set[v], self.output_vectors_testing_set[v])

            self.log_prediction(self.vector_date_list_testing_set[v], self.input_vectors_testing_set[v],
                                self.output_vectors_testing_set[v], learned_from=False)

            self.reset_i_o_de()

            #sleep(0.005)

    def do_get_info(self):
        '''Get latest info on the network at a glance. Store in self.info.'''
        self.info = ['Name: ' + self.name,
                'Security: ' + self.security,
                'Input Vector Parameters: ' + ','.join(self.input_vector_parameters),
                'Output Vector Parameters: ' + ','.join(str(self.output_vector_parameters)),
                'Training/Testing Ratio: ' + str(self.training_testing_ratio),
                'Learning Rate: ' + str(self.learning_rate),
                'Neuron Activation Function: ' + str(self.neuron_activation_function.__name__),
                'Effector Activation Function: ' + str(self.effector_activation_function.__name__),
                'Receptor Matrix Dimensions (r X c): ' + str(self.network_dimensions[0][0]) + \
                ' X ' + str(self.network_dimensions[0][1]),
                'Neuron Matrix Dimensions (r X c): ' + str(self.network_dimensions[1][0]) + \
                ' X ' + str(self.network_dimensions[1][1]),
                'Effector Matrix Dimensions (r X c): ' + str(self.network_dimensions[2][0]) + \
                ' X ' + str(self.network_dimensions[2][1]),
                'Trained on Function: ' + str(self.trained_on_function),
                'Trained on Security ' + str(self.trained_on_security),
                'Optimized: ' + str(self.optimized)]

        return self.info

    def forward_propagate(self, input_vector, output_vector):
        '''Predict the output(s) of the network using the given input_vector. Assumes the network has already been
        built with weight, input, output, and delta_error arrays.'''

        self.inputs[0] = np.array([input_vector]).T
        self.outputs[0] = np.array(self.neuron_activation_function(np.array(self.inputs[0]).T, deriv=False)).T
        temp = []
        for neuron_index in range(len(self.weights[0])):  # iterate over each neuron
            temp.append(self.outputs[0][neuron_index][0] * np.array(self.weights[0][neuron_index]))  # multiply neuron's charge by each of its weights for the following layer
        self.inputs[1] = np.array([sum(temp)]).T  # sum column, transpose, set as next layer's input

        for layer_index in range(1, len(self.inputs)-1):    # only go up to the second to last input in loop

            self.outputs[layer_index] = np.array(self.neuron_activation_function(np.array(self.inputs[layer_index]).T, deriv=False)).T
            temp = []

            for neuron_index in range(len(self.weights[layer_index])):    # iterate over each neuron

                temp.append(self.outputs[layer_index][neuron_index][0] * np.array(self.weights[layer_index][neuron_index]))    # multiply neuron's charge by each of its weights for the following layer

            self.inputs[layer_index + 1] = np.array([sum(temp)]).T    # sum column, transpose, set as next layer's input

        self.outputs[len(self.outputs) - 1] = self.effector_activation_function(self.inputs[len(self.outputs) - 1], deriv=False)

        # calculate errors in outputs
        neuron_index = 0
        self.vector_output_error = []
        for expected_output in output_vector:  # calculate delta_error for effector neurons

            self.vector_output_error.append(
                [((expected_output - self.outputs[len(self.outputs) - 1][neuron_index][0]) / expected_output)])

            neuron_index += 1

        if self.vector_output_error == []:  # expected outputs that have yet to come, will yield empty error value
            self.vector_output_error = [[0]]

    def back_propagate(self, output_vector):
        '''Back propagate the output error. Assumes an input has already been forward propagated.'''

        # calculate delta_errors
        neuron_index = 0
        for expected_output in output_vector:  # calculate delta_error for effector neurons

            self.delta_errors[len(self.delta_errors) - 1][neuron_index][0] = \
                (self.effector_activation_function(expected_output,deriv=False) - self.outputs[len(self.outputs) - 1][neuron_index][0]) * \
                self.effector_activation_function(self.inputs[len(self.inputs) - 1][neuron_index][0], deriv=True)
            neuron_index += 1

        for layer_index in reversed(range(len(self.delta_errors) - 1)):

            for neuron_index in range(len(self.weights[layer_index])):
                # multiply delta errors of following layer with weights of each neuron in current layer and sum to
                # get delta error of neuron in current layer
                a = np.array(np.array(self.delta_errors[layer_index + 1]).T) * np.array(
                    self.weights[layer_index][neuron_index])
                b = sum(a[0]) * self.neuron_activation_function(self.inputs[layer_index][neuron_index], deriv=True)
                self.delta_errors[layer_index][neuron_index][0] = b

        # update weights
        for layer_index in range(len(self.inputs) - 1):

            weight_changes = np.array(np.array(self.delta_errors[layer_index]).T) * \
                             self.learning_rate * \
                             np.array(np.array(self.inputs[layer_index]).T)

            for neuron_index in range(len(self.weights[layer_index])):

                for weight_index in range(len(self.weights[layer_index][neuron_index])):
                    self.weights[layer_index][neuron_index][weight_index] = \
                        self.weights[layer_index][neuron_index][weight_index] + weight_changes[0][neuron_index]

    def log_prediction(self, date_vector, input_vector, output_vector, learned_from=False):
        '''Log prediction information: input vector, output vector, prediction start date,'''

        DIGITS_TO_ROUND_OFF = 6

        # STORE THE RESULTS IN THE PREDICTION LOG
        if len(self.prediction_log['epoch']) > 1:
            new_epoch = int(self.prediction_log['epoch'][len(self.prediction_log['epoch']) - 1]) + 1
        else:
            new_epoch = 1
        self.prediction_log['epoch'].append(new_epoch)

        learned_from = learned_from
        self.prediction_log['learned_from'].append(learned_from)

        self.prediction_log['security'].append(self.security)

        Prediction_Start_Date = date_vector
        self.prediction_log['prediction_start_date'].append(Prediction_Start_Date)

        Input_Vector = np.around(input_vector, DIGITS_TO_ROUND_OFF).tolist()
        self.prediction_log['input_vector'].append(Input_Vector)

        Expected_Output_Vector = np.around(output_vector, DIGITS_TO_ROUND_OFF).tolist()
        self.prediction_log['expected_output_vector'].append((Expected_Output_Vector))

        Output_Vector_Dates = []
        for period in self.output_vector_parameters:
            Output_Vector_Dates.append(date_vector + timedelta(days=period))
        self.prediction_log['prediction_end_dates'].append(Output_Vector_Dates)

        Actual_Output_Vector = [np.around(self.outputs[len(self.outputs) - 1][x][0], DIGITS_TO_ROUND_OFF) for x in range(len(self.outputs[len(self.outputs) - 1]))]
        self.prediction_log['actual_output_vectors'].append(Actual_Output_Vector)

        Errors = np.around(np.array(self.vector_output_error).T[0], DIGITS_TO_ROUND_OFF).tolist()
        self.prediction_log['errors'].append(Errors)

        return True

    def do_plot_outputs(self):
        '''Generate and return a graph from the error value(s) recorded for each input vector, along with the
        predicted and expected values.'''

        if self.graphed_outputs:

            for series_index in range(len(self.prediction_log['expected_output_vector'][0])):
                y = []
                for i in range(len(self.prediction_log['expected_output_vector'])):
                    try:    # handle case of empty set when the data point is not available
                        y.append(self.prediction_log['expected_output_vector'][i][series_index])
                    except:
                        y.append(0)
                x = [list(x) for x in zip(*self.prediction_log['prediction_end_dates'])][series_index]
                plt.plot(x,y,label='expected_output_'+str(self.output_vector_parameters[series_index]))

            for series_index in range(len(self.prediction_log['actual_output_vectors'][0])):
                y = []
                for i in range(len(self.prediction_log['actual_output_vectors'])):
                    y.append(self.prediction_log['actual_output_vectors'][i][series_index])
                x = (np.array(self.prediction_log['prediction_end_dates']).T)[series_index]
                plt.plot(x,y,label='actual_output_'+str(self.output_vector_parameters[series_index]))

            plt.subplot(111).legend()

            self.do_export_plot(plt, self.short_info)   # export plot to folder in self.save_path '/plots/'
            plt.close()  # close the plot instance, so future plots do not graph on top of it

            return True

        else:

            return False

    def do_optimize(self):
        '''Optimize the network construction by varying the number of rows, columns, learning rate, and the
        training to testing data ratio.'''

        learning_rate_list = np.linspace(0.02, 0.05, 2)
        neuron_matrix_rows_list = np.linspace(2, 4, 1)
        neuron_matrix_columns_list = np.linspace(2, 4, 1)
        training_testing_ratio_list = np.linspace(0.4, 0.8, 2)
        self.optimizer_median_errors_dict = {}

        optimization_count = len(learning_rate_list) * len(neuron_matrix_columns_list) * \
                             len(neuron_matrix_columns_list) * len(training_testing_ratio_list)

        with tqdm(range(optimization_count), desc='OPTIMIZATION') as opBar:

            for lr in trange(len(learning_rate_list), desc='learning rate'):

                for r in trange(len(neuron_matrix_rows_list), desc='neuron rows'):

                    for c in trange(len(neuron_matrix_columns_list), desc='neuron columns'):

                        for rat in trange(len(training_testing_ratio_list), desc='tt ratio'):

                            self.do_set_network_parameters(name=self.name,
                                            security=self.security,
                                            training_testing_ratio=training_testing_ratio_list[rat],
                                            learning_rate=learning_rate_list[lr],
                                            input_vector_parameters=self.input_vector_parameters,
                                            neuron_matrix_rows=int(neuron_matrix_rows_list[r]),
                                            neuron_matrix_columns=int(neuron_matrix_columns_list[c]),
                                            output_vector_parameters=self.output_vector_parameters,
                                            neuron_activation_function=self.neuron_activation_function,
                                            effector_activation_function=self.effector_activation_function)
                            self.build()
                            if self.trained_on_function:
                                self.do_train_on_function()
                            else:
                                self.do_train_on_security()
                            self.optimizer_median_errors_dict[
                                str(np.median(self.prediction_log['errors']))] = self.short_info

                            #sleep(0.005)

                            opBar.update(1)

        optimal_values_string = self.optimizer_median_errors_dict[min(self.optimizer_median_errors_dict)].split('##')[1].split()

        opt_nrows = int(optimal_values_string[0])
        opt_ncols = int(optimal_values_string[1])
        opt_ratio = float(optimal_values_string[2])
        opt_rate = float(optimal_values_string[3])

        print('Optimal Values:')
        print('  Neuron Rows: ' + str(opt_nrows))
        print('  Neuron Columns: ' + str(opt_ncols))
        print('  Training/Testing Ratio: ' + str(opt_ratio))
        print('  Learning Rate: ' + str(opt_rate))

        print('Applying settings...')

        # apply the optimal settings
        self.do_set_network_parameters(name=self.name,
                                    security=self.security,
                                    training_testing_ratio=opt_ratio,
                                    learning_rate=opt_rate,
                                    input_vector_parameters=self.input_vector_parameters,
                                    neuron_matrix_rows=opt_nrows,
                                    neuron_matrix_columns=opt_ncols,
                                    output_vector_parameters=self.output_vector_parameters,
                                    neuron_activation_function=self.neuron_activation_function,
                                    effector_activation_function=self.effector_activation_function)
        self.build()
        if self.trained_on_function:
            self.do_train_on_function()
        else:
            self.do_train_on_security()

        print('finished applying optimal settings.')

        return True


def sigmoid(x, deriv=False):
    '''Sigmoid function and its derivative. Return the matrix after each element has passed through function
    element-wise. Use numpy. '''

    if deriv:  # if derivation of activation function requested
        return (np.exp(-x)/(1+np.exp(-x))**2)

    else:
        return 1 / (1 + np.exp(-x))


def tanh(x, deriv=False):
    '''Hyperbolic Tan function and its derivative. Return matrix of element-wise processed numbers.'''

    if deriv:  # if derivation of activation function requested
        return (1 / np.cosh(x)) ** 2

    else:
        return np.tanh(x)


def linear(x, deriv=False):
    '''Compute the linear function, which is just the same values as are inputted.'''

    if deriv:  # if derivation of activation function requested.
        return 1

    else:
        return x


def isfloat(value):
  '''check if value is a float.'''
  try:
    float(value)
    return True
  except ValueError:
    return False

if __name__ == '__main__':
    app = Network_Builder()
    app.prompt = '>> '
    app.cmdloop()
