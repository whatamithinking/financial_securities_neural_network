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

class FSNN:
    '''Control the Neural Network and input data.'''

    def __init__(self, parent=None):
        '''Collect inputs from the user to decide whether to load an existing network or make a new one and what
        the characteristics of a new network might be.'''

        security_parameters = ['Close', 'Open', 'Volume', 'Adj Close',
                               '5dma', '10dma', '20dma', '50dma',
                               '100dma', '200dma']

        network_action = ''
        network_action = raw_input('Would you like to create a new network ("n") or load an existing one ("l")? ')

        if network_action.strip() == 'n':
            network_settings_accurate = 'N'
            while network_settings_accurate == 'N':
                if raw_input('Default(d) or new(n) settings? ').lower() == 'd':
                    name = 'yhoo_fsnn'
                    security = 'V'
                    rows = 3
                    columns = 3
                    training_testing_ratio = 0.7
                    input_vector_parameters = ['Close', '50dma', '200dma']
                    outputs = [5, 50, 100, 200]
                    learning_rate = 0.05
                    neuron_activation_function_name = 'sigmoid'
                    neuron_activation_function = getattr(sys.modules[__name__], neuron_activation_function_name)
                    effector_activation_function_name = 'linear'
                    effector_activation_function = getattr(sys.modules[__name__], effector_activation_function_name)
                else:
                    name = raw_input('Name your network: ')

                    good_ticker = False
                    while not good_ticker:
                        security = raw_input('Ticker of security to train network on: ').upper()
                        try:
                            temp_data = web.DataReader(security, 'yahoo', '2010/01/01',date.today())['Close']
                            good_ticker = True
                        except:
                            print('Could not find ticker.')

                    rows = int(raw_input('Number of rows of Neurons: '))
                    columns = int(raw_input('Number of columns of Neurons: '))
                    training_testing_ratio = float(raw_input('Training / Testing ratio (decimal): '))
                    print('')
                    print('The following input vector parameters are available: ')
                    print('')
                    for parameter in security_parameters:
                        print(parameter)
                    print('')
                    input_vector_parameters = raw_input('List the input vector parameters: ').split()
                    print('')
                    outputs = [int(x) for x in
                               raw_input('Number of days to lookforward (starting day after prediction): ').split()]
                    print('')
                    learning_rate = float(raw_input('Learning coefficient: '))
                    print('Activation Functions: sigmoid, tanh, or linear')
                    neuron_activation_function_name = raw_input(
                        'Which activation function would you like to use for NEURONS?: ')
                    neuron_activation_function = getattr(sys.modules[__name__], neuron_activation_function_name)
                    effector_activation_function_name = raw_input(
                        'Which activation function would you like to use for EFFECTORS?: ')
                    effector_activation_function = getattr(sys.modules[__name__], effector_activation_function_name)

                print('*** Here are the network settings you have selected ***')
                print('Name: %s' % name)
                print('Security: %s' % security)
                print('Rows: %d' % rows)
                print('Columns: %d' % columns)
                print('Training/Testing Ratio: %f' % training_testing_ratio)
                print('Output Days (counting day of prediction): ')
                for period in outputs:
                    print('     %s' % period)
                print('')
                print('Input Vector Parameters: ')
                for selected_parameter in input_vector_parameters:
                    print('     %s' % selected_parameter)
                print('Learning Coefficient: ' + str(learning_rate))
                print('NEURON Activation Function: ' + neuron_activation_function_name)
                print('EFFECTOR Activation Function: ' + effector_activation_function_name)
                print('*** End network parameters ***')
                network_settings_accurate = raw_input('Are these settings correct? (Y/N) ')

            self.network = Network(name=name, security=security, rows=rows, columns=columns,
                                   training_testing_ratio=training_testing_ratio,
                                   output_vector_parameters=outputs, input_vector_parameters=input_vector_parameters,
                                   neuron_activation_function=neuron_activation_function,
                                   effector_activation_function=effector_activation_function,
                                   learning_rate=learning_rate)

            self.network.graph_error_over_time()

            self.save_network()

        elif network_action.strip() == 'load':

            self.load_network()

    def load_network(self):
        '''Load network from file.'''

        network_location = input("Please type the address of the network's file: ")
        self.network = pickle.load(open(network_location, 'rb'))

        return True

    def save_network(self):
        '''Save the Network object using pickle library.'''
        file_name = raw_input('Name file: ')
        save_location = raw_input('Type file save location: ')
        save_path = save_location + '/' + file_name + '.p'
        print('Saving network...')
        pickle.dump(self.network, open(save_path, 'wb'))
        print('Network saved successfully!')
        return True


class Network:
    '''The Neural Network houses all of the Receptor, Neuron, and Effector objects.'''

    SECURITY_START_DATE = datetime(year=1900, month=1, day=1)
    SECURITY_END_DATE = datetime(date.today().year,date.today().month,date.today().day)

    def __init__(self, neuron_activation_function,effector_activation_function, name='SNP_ANN_FSNN', security='SNP',
                 rows=50, columns=2, training_testing_ratio=0.7,
                 output_vector_parameters=[1, 4], input_vector_parameters=['Close', 'Volume', 'Open', 'Adj Close'],
                 learning_rate=0.05):

        print('Initializing network...')

        self.security = security
        self.name = name
        self.training_testing_ratio = training_testing_ratio
        self.output_vector_parameters = output_vector_parameters
        self.output_vector_parameters_count = len(output_vector_parameters)
        self.input_vector_parameters = input_vector_parameters
        self.input_vector_parameters_count = len(input_vector_parameters)

        if neuron_activation_function == None:
            self.neuron_activation_function = self.sigmoid
        else:
            self.neuron_activation_function = neuron_activation_function
        if effector_activation_function == None:
            self.effector_activation_function = self.linear
        else:
            self.effector_activation_function = effector_activation_function
        self.learning_rate = learning_rate

        self.receptor_matrix_rows = np.size([0])  # 1 column for the input vectors
        self.receptor_matrix_columns = len(self.input_vector_parameters)
        self.neuron_matrix_rows = rows
        self.neuron_matrix_columns = columns
        self.effector_matrix_rows = len(self.output_vector_parameters)
        self.effector_matrix_columns = np.size([0])  # 1 column of output vectors

        self.input_vectors = None
        self.output_vectors = None
        self.vector_date_list = None
        self.input_vectors_training_set = None
        self.input_vectors_testing_set = None
        self.output_vectors_training_set = None
        self.output_vectors_testing_set = None
        self.vector_date_list_training_set = None
        self.vector_date_list_testing_set = None

        self.weights = []   # The weights for each layer
        self.outputs = []
        self.inputs = []
        self.delta_errors = []
        self.vector_output_error = []

        self.prediction_log = {'epoch':[],'learned_from':[],'security':[],'prediction_start_date':[],'input_vector':[],
                               'expected_output_vector':[],'prediction_end_dates':[],'actual_output_vectors':[],
                               'errors':[]}  # store all of the predictions the network ever makes
        print('epoch', 'learned_from', 'security', 'prediction_start_date', 'input_vector', 'expected_output_vector',
         'prediction_end_dates', 'actual_output_vector', 'errors')

        # STORE DIMENSIONS OF NETWORK
        self.network_dimensions = [[self.receptor_matrix_rows, self.receptor_matrix_columns],
                              [self.neuron_matrix_rows, self.neuron_matrix_columns],
                              [self.effector_matrix_rows, self.effector_matrix_columns]]

        # GENERATE THE EMPTY NETWORK
        self.build()

        # TRAIN NETWORK
        self.train()

    def generate_output_vectors(self):
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

    def generate_input_vectors(self):
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

    def generate_vector_date_list(self):
        '''Generate the list of dates corresponding to the vector inputs and outputs.'''

        print('     generating vector date list...')

        vector_date_list = web.DataReader(self.security, 'yahoo', self.SECURITY_START_DATE, self.SECURITY_END_DATE)[
            'Close'].index

        return vector_date_list

    def fetch_training_testing_vectors(self):
        '''Get all of the data for training and for testing, and chop according in the ratio provided by the user.'''

        print('     splicing training/testing data vectors...')

        self.input_vectors = self.generate_input_vectors()  # get list of input vectors
        self.output_vectors = self.generate_output_vectors()
        self.vector_date_list = self.generate_vector_date_list()

        training_end_index = int(self.training_testing_ratio * (len(self.input_vectors)-1))

        self.input_vectors_training_set = [self.input_vectors[x] for x in range(training_end_index)]
        self.input_vectors_testing_set = [self.input_vectors[training_end_index + x] for x in
                                          range(len(self.input_vectors)-training_end_index-1)]

        self.output_vectors_training_set = [self.output_vectors[x] for x in range(training_end_index)]
        self.output_vectors_testing_set = [self.output_vectors[training_end_index + x] for x in
                                           range(len(self.output_vectors)-training_end_index-1)]

        self.vector_date_list_training_set = [self.vector_date_list[x] for x in range(training_end_index)]
        self.vector_date_list_testing_set = [self.vector_date_list[training_end_index + x] for x in
                                        range(len(self.vector_date_list)-training_end_index-1)]

        return True

    def build(self):
        '''The neural network requires the input and output of every neuron to be saved. All of the inputs and outputs
        are saved in separate matrices to improve readability.'''

        print('     building empty weight matrices...')
        # matrix of weights for inputs. Number of rows equals number of input vectors. Number of columns is 1.
        self.weights.append(np.random.rand(self.network_dimensions[0][1], np.size([0])))
        # matrix of weights for hidden layer. Number of rows/cols = number of rows of neurons.
        # Each row is a neuron's weights for the following layer. Last weight array for effectors has different shape.
        for x in range(0,self.network_dimensions[1][1] - 1):
            self.weights.append(np.random.rand(self.network_dimensions[1][0], self.network_dimensions[1][0]))
        self.weights.append(np.random.rand(self.network_dimensions[1][0], self.network_dimensions[2][0]))

        print('     building empty input/output/delta_error matrices...')
        self.reset_i_o_de()

        return True

    def reset_i_o_de(self):

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

    def train(self):
        '''Train the neural network, ANN or RC, using the split training and testing data.'''

        self.fetch_training_testing_vectors()

        print('')
        print('Start Training')
        print('')

        for i in range(3):

            for v in range(len(self.vector_date_list_training_set)):

                self.forward_propagate(self.input_vectors_training_set[v], self.output_vectors_training_set[v])

                self.back_propagate(self.output_vectors_training_set[v])

                self.log_prediction(self.vector_date_list_training_set[v], self.input_vectors_training_set[v],
                                    self.output_vectors_training_set[v], learned_from=True)

                self.reset_i_o_de()

        print('')
        print('End Training')
        print('')
        print('Start Testing')
        print('')

        for v in range(len(self.vector_date_list_testing_set)):

            self.forward_propagate(self.input_vectors_testing_set[v], self.output_vectors_testing_set[v])

            self.log_prediction(self.vector_date_list_testing_set[v], self.input_vectors_testing_set[v],
                                self.output_vectors_testing_set[v], learned_from=False)

            self.reset_i_o_de()

        print('')
        print('End Testing')
        print('')

    def get_info(self):
        '''Return info on the network at a glance.'''
        return ['Name: ' + self.name, 'Security: ' + self.security,
                'Input Vector Parameters: ' + ','.join(self.input_vector_parameters),
                'Output Vector Parameters: ' + ','.join(str(self.output_vector_parameters)),
                'Training/Testing Ratio: ' + str(self.training_testing_ratio),
                'Average_Prediction_Error: ' + str(self.average_prediction_error),
                'Current Training Error: ' + str(self.current_training_error),
                'Epochs Trained On: ' + str(self.epochs_trained_on),
                'Receptor Matrix Dimensions (r X c): ' + str(self.receptor_matrix_rows) + \
                ' X ' + str(self.receptor_matrix_columns),
                'Neuron Matrix Dimensions (r X c): ' + str(self.neuron_matrix_rows) + ' X ' + \
                str(self.neuron_matrix_columns),
                'Effector Matrix Dimensions (r X c): ' + str(self.effector_matrix_rows) + ' X ' + \
                str(self.effector_matrix_columns)]

    def forward_propagate(self, input_vector, output_vector):
        '''Predict the output(s) of the network using the given input_vector. Assumes the network has already been
        built with weight, input, output, and delta_error arrays.'''

        # FEED FORWARD
        self.outputs[0] = self.inputs[0] = np.array([input_vector]).T
        input_sum = sum(np.array(np.array(self.weights[0]).T) * np.array(np.array(self.inputs[0]).T)[0])[0]
        self.inputs[1] = np.array([[input_sum] for x in range(0,self.network_dimensions[1][0])])

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

        if self.vector_output_error == []:
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
        self.prediction_log['expected_output_vector'].append(Expected_Output_Vector)

        Output_Vector_Dates = []
        for period in self.output_vector_parameters:
            Output_Vector_Dates.append(date_vector + timedelta(days=period))
        self.prediction_log['prediction_end_dates'].append(Output_Vector_Dates)

        Actual_Output_Vector = [np.around(self.outputs[len(self.outputs) - 1][x][0], DIGITS_TO_ROUND_OFF) for x in range(len(self.outputs[len(self.outputs) - 1]))]
        self.prediction_log['actual_output_vectors'].append(Actual_Output_Vector)

        Errors = np.around(self.vector_output_error[0], DIGITS_TO_ROUND_OFF).tolist()
        self.prediction_log['errors'].append(Errors)

        print(new_epoch, learned_from, self.security, Prediction_Start_Date, Input_Vector, Expected_Output_Vector,
              Output_Vector_Dates, Actual_Output_Vector, Errors)

    def predict_and_learn(self, input_vector):
        '''Take in the input_vector, predict an output and learn from that one input vector, by altering the
        weights in the network.'''

    def graph_error_over_time(self):
        '''Generate and return a graph from the error value(s) recorded for each input vector, along with the
        predicted and expected values.'''

        plt.ion()   # enable interactive plotting

        for series_index in range(len(self.prediction_log['expected_output_vector'])):
            y = [list(x) for x in zip(*self.prediction_log['expected_output_vector'])][series_index]
            x = [list(x) for x in zip(*self.prediction_log['prediction_end_dates'])][series_index]
            plt.plot(x,y,label='expected_output_'+str(self.output_vector_parameters[series_index]))

        for series_index in range(len(self.prediction_log['actual_output_vectors'])):
            y = (np.array(self.prediction_log['actual_output_vectors']).T)[series_index]
            x = (np.array(self.prediction_log['prediction_end_dates']).T)[series_index]
            plt.plot(x,y,label='actual_output_'+str(self.output_vector_parameters[series_index]))

        plt.subplot(111).legend()
        plt.show()

        return True

def sigmoid(x, deriv=False):
    '''Sigmoid function and its derivative. Return the matrix after each element has passed through function
    element-wise. Use numpy. '''

    if deriv:  # if derivation of activation function requested
        return sigmoid(x) * (1 - sigmoid(x))

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

if __name__ == '__main__':
    FSNN()