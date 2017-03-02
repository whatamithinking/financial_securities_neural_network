'''
Program: Security Trading Using Neural Networks
Author: Connor S. Maynes
Purpose: Generate neural networks from security data and predict future prices.
'''

import pandas.io.data as web  # Package and modules for importing data; this code may change depending on pandas version
from datetime import date, datetime, timedelta
import numpy as np
import math
import pickle


class STUNN:
    '''Control the Neural Network and input data.'''

    def __init__(self, parent=None):
        '''Collect inputs from the user to decide whether to load an existing network or make a new one and what
        the characteristics of a new network might be.'''

        security_parameters = ['Close', 'Open', 'Volume', 'Adj Close',
                               '5day_moving_avg', '10day_moving_avg', '20day_moving_avg', '50day_moving_avg',
                               '100day_moving_avg', '200day_moving_avg']

        network_action = ''
        network_action = raw_input('Would you like to create a new network ("new") or load an existing one ("load")? ')

        if network_action.strip() == 'new':
            network_settings_accurate = 'N'

            while network_settings_accurate == 'N':

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
                learning_coefficient = float(raw_input('Learning coefficient: '))
                print('Activation Functions: sigmoid, tanh, or linear')
                receptor_activation_function_name = raw_input(
                    'Which activation function would you like to use for RECEPTORS?: ')
                receptor_activation_function = getattr(STUNN, receptor_activation_function_name)
                neuron_activation_function_name = raw_input(
                    'Which activation function would you like to use for NEURONS?: ')
                neuron_activation_function = getattr(STUNN, neuron_activation_function_name)
                effector_activation_function_name = raw_input(
                    'Which activation function would you like to use for EFFECTORS?: ')
                effector_activation_function = getattr(STUNN, effector_activation_function_name)

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
                print('Learning Coefficient: ' + str(learning_coefficient))
                print('RECEPTOR Activation Function: ' + receptor_activation_function_name)
                print('NEURON Activation Function: ' + neuron_activation_function_name)
                print('EFFECTOR Activation Function: ' + effector_activation_function_name)
                print('*** End network parameters ***')
                network_settings_accurate = raw_input('Are these settings correct? (Y/N) ')

            self.network = Network(name=name, security=security, rows=rows, columns=columns,
                                   training_testing_ratio=training_testing_ratio,
                                   output_vector_parameters=outputs, input_vector_parameters=input_vector_parameters,
                                   receptor_activation_function=receptor_activation_function,
                                   neuron_activation_function=neuron_activation_function,
                                   effector_activation_function=effector_activation_function,
                                   learning_coefficient=learning_coefficient)

        elif network_action.strip() == 'load':

            network_location = input("Please type the address of the network's file: ")
            self.network = pickle.load(open(network_location, 'rb'))

    @staticmethod
    def sigmoid(value, deriv=False):
        '''Sigmoid function and its derivative. Return the matrix after each element has passed through function
        element-wise. '''

        if deriv:  # if derivation of activation function requested
            return math.exp(-1 * value) / math.pow(1 + math.exp(-1 * value), 2)

        else:
            return 1/(1+math.exp(-1 * value))

    @staticmethod
    def tanh(value, deriv=False):
        '''Hyperbolic Tan function and its derivative. Return matrix of element-wise processed numbers.'''

        if deriv:  # if derivation of activation function requested
            return math.pow(math.pow(np.cosh(value), -1), 2)

        else:
            return np.tanh(value)

    @staticmethod
    def linear(value, deriv=False):
        '''Compute the linear function, which is just the same values as are inputted.'''

        if deriv:  # if derivation of activation function requested.
            return 1

        else:
            return value

    def save_network(self):
        '''Save the Network object using pickle library.'''
        file_name = input('Name file: ')
        save_location = input('Type file save location: ')
        save_path = save_location + '/' + file_name + '.p'
        pickle.dump(self.network, open(save_path, 'wb'))
        return True


class Network:
    '''The Neural Network houses all of the Receptor, Neuron, and Effector objects.'''

    SECURITY_START_DATE = (datetime(year=1900, month=1, day=1))
    SECURITY_END_DATE = datetime(date.today().year,date.today().month,date.today().day)

    def __init__(self, name='SNP_ANN_STUNN', security='SNP', rows=50, columns=2, training_testing_ratio=0.7,
                 output_vector_parameters=[1, 4], input_vector_parameters=['Close', 'Volume', 'Open', 'Adj Close'],
                 receptor_activation_function=STUNN.linear, neuron_activation_function=STUNN.sigmoid,
                 effector_activation_function=STUNN.linear, learning_coefficient=0.05):
        '''Get inputs from the user regarding what they want to do. If the user wants to load a previously saved
        ANN by this program, they will be asked for the file. If the user wants to make a new neural network,
        they can input what type of ANN they want, their data (CSV format with each row as a vector that
        contains the inputs/outputs they wish to predict), the dimensions of the NN (or number of neurons if RC),
        the inputs to include (comma-separated) (assume first row of csv contains the names), the outputs to predict,
        the lookforward period (how many vectors out from this one the user wants to predict for the outputs --
        may have multiple periods), the activation function to use in the hidden layers, and the training to testing
        data ratio.'''

        self.security = security
        self.name = name
        self.training_testing_ratio = training_testing_ratio
        self.output_vector_parameters = output_vector_parameters
        self.output_vector_parameters_count = len(output_vector_parameters)
        self.input_vector_parameters = input_vector_parameters
        self.input_vector_parameters_count = len(input_vector_parameters)

        self.receptor_activation_function = receptor_activation_function
        self.neuron_activation_function = neuron_activation_function
        self.effector_activation_function = effector_activation_function
        self.learning_coefficient = learning_coefficient

        self.average_prediction_error = 0
        self.current_training_error = 0
        self.epochs_trained_on = 0

        self.receptor_matrix_rows = len(self.input_vector_parameters)
        self.receptor_matrix_columns = np.size([0])  # 1 column for the input vectors
        self.neuron_matrix_rows = rows
        self.neuron_matrix_columns = columns
        self.effector_matrix_rows = len(self.output_vector_parameters)
        self.effector_matrix_columns = np.size([0])  # 1 column of output vectors

        self.prediction_log = []  # store all of the predictions the network ever makes
        '''Format: [Epoch, Learned_From=T/F, Prediction_Start_Date, Input_Vector, Expected_Output_Vector,
            Actual_Output_Vector, Errors]'''
        self.prediction_log.append(['epoch', 'learned_from', 'prediction_start_date', 'input_vector', 'expected_output_vector',
                 'actual_output_vector', 'errors'])
        print(self.prediction_log[0])

        # STORE DIMENSIONS OF NETWORK
        network_dimensions = [[self.receptor_matrix_rows, self.receptor_matrix_columns],
                              [self.neuron_matrix_rows, self.neuron_matrix_columns],
                              [self.effector_matrix_rows, self.effector_matrix_columns]]
        self.network_dimensions = network_dimensions

        # GENERATE THE EMPTY NETWORK
        self.network = self.build()

        # TRAIN NETWORK
        self.train()

    def generate_output_vectors(self):
        '''Generate all of the output vectors, using the user's inputted output vectors.'''

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
        list of input vectors.'''

        input_vector_list = []
        temp = []

        for parameter in self.input_vector_parameters:

            if 'moving_avg' in parameter:
                ma_range = int(parameter.split('d', 1)[0])  # get number of days for moving average
                close_prices = web.DataReader(self.security, 'yahoo', self.SECURITY_START_DATE, self.SECURITY_END_DATE)['Close']
                ma = np.round(close_prices.rolling(ma_range, False).mean(),2)
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

        vector_date_list = web.DataReader(self.security, 'yahoo', self.SECURITY_START_DATE, self.SECURITY_END_DATE)[
            'Close'].index

        return vector_date_list

    def split_training_and_testing_data(self):
        '''Split apart the training and testing data using the information given by the user when they built the
        NN. Use the user's given training to testing ratio. Output the input vector training set, input vector testing
        set, the output vector training set, and the output vector testing set in a list.'''

        training_end_index = 0
        training_end_index = int(np.rint(self.training_testing_ratio * (len(self.input_vectors)-1)))

        self.input_vectors_training_set = [self.input_vectors[x] for x in range(training_end_index)]
        self.input_vectors_testing_set = [self.input_vectors[training_end_index + x] for x in range(len(self.input_vectors)-training_end_index-1)]

        self.output_vectors_training_set = [self.output_vectors[x] for x in range(training_end_index)]
        self.output_vectors_testing_set = [self.output_vectors[training_end_index + x] for x in range(len(self.output_vectors)-training_end_index-1)]

        self.vector_date_list_training_set = [self.vector_date_list[x] for x in range(training_end_index)]
        self.vector_date_list_testing_set = [self.vector_date_list[training_end_index + x] for x in
                                        range(len(self.vector_date_list)-training_end_index-1)]

        return True

    def build(self):
        '''Build the Neural Network from the user's inputs. Create a matrix of Receptors
        with a count equal to the number of input vectors entered by the user, a matrix of neurons with the
        dimensions specified by the user, and a matrix of Effectors, with each column being a different lookforward
        period and each row a different value to predict. Set flag in each Neuron to indicate which activation function
        to use. Each column of neurons is connected to the following column of neurons. Only the first column of neurons
        is connected to the Receptor matrix. Only the last column of neurons is connected to the Effector matrix. The
        Receptor, Neuron, and Effector Matricies become the three columns of the Network. Return the network.'''

        receptor_matrix = [
            [Receptor(charge=0, receptor_location=[row, col], network_dimensions=self.network_dimensions,
                      activation_function=self.receptor_activation_function,
                      learning_coefficient=self.learning_coefficient) for col
             in
             range(self.receptor_matrix_columns)] for row in range(self.receptor_matrix_rows)]

        neuron_matrix = [
            [Neuron(charge=0, neuron_location=[row, col], network_dimensions=self.network_dimensions,
                    activation_function=self.neuron_activation_function, learning_coefficient=self.learning_coefficient)
             for col in
             range(self.neuron_matrix_columns)] for row in range(self.neuron_matrix_rows)]

        effector_matrix = [
            [Effector(charge=0, effector_location=[row, col], expected_charge=0, activation_function=STUNN.linear,
                      learning_coefficient=self.learning_coefficient) for
             col in range(self.effector_matrix_columns)]
            for row in
            range(self.effector_matrix_rows)]

        return [receptor_matrix, neuron_matrix, effector_matrix]

    def train(self):
        '''Train the neural network, ANN or RC, using the split training and testing data.'''

        # GET THE DATA AS INPUT VECTORS FOR THE SECURITY AND PARSE FOR TRAINING AND TESTING
        self.input_vectors = self.generate_input_vectors()  # get list of input vectors
        self.output_vectors = self.generate_output_vectors()
        self.vector_date_list = self.generate_vector_date_list()
        self.split_training_and_testing_data()

        for v in range(len(self.vector_date_list_training_set)):

            input_vector_for_date = self.input_vectors_training_set[v]
            output_vector_for_date = self.output_vectors_training_set[v]

            # CHARGE EFFECTORS WITH OUTPUT VECTOR FOR DATE
            for col in range(self.network_dimensions[2][1]):

                for row in range(self.network_dimensions[2][0]):
                    self.network[2][row][col].set_expected_charge(output_vector_for_date[row])

            # CHARGE RECEPTORS WITH INPUT VECTOR FOR DATE. *** FIRE RECEPTORS TO NEURONS.
            for col in range(self.network_dimensions[0][1]):

                for row in range(self.network_dimensions[0][0]):
                    self.network[0][row][col].set_charge(input_vector_for_date[row])  # set receptor's charge
                    self.network = self.network[0][row][col].fire(self.network)  # fire and get network with changes

            # FEEDFORWARD INPUTS. *** FIRE BETWEEN NEURONS AND TO EFFECTORS
            for col in range(self.network_dimensions[1][1]):

                for row in range(self.network_dimensions[1][0]):
                    self.network = self.network[1][row][col].fire(self.network)  # fire each neuron

            # CALCULATE DELTA ERRORS THROUGH BACKPROPAGATION FOR EFFECTORS, NEURONS, and RECEPTORS
            output_error_list = []
            output_list = []
            for matrix_index in reversed(range(0, 3)):

                for col in reversed(range(self.network_dimensions[matrix_index][1])):

                    for row in range(self.network_dimensions[matrix_index][0]):

                        if matrix_index <= 1:
                            self.network[matrix_index][row][col].calculate_delta_error(self.network)

                        else:
                            self.network[matrix_index][row][col].calculate_delta_error()
                            output_error_list.append(self.network[matrix_index][row][col].error)
                            output_list.append(self.network[matrix_index][row][col].activated_charge)

            # ADJUST WEIGHTS OF NEURONS, RECEPTORS USING DELTA ERROR VALUES. CHARGES ARE RESET AUTOMATICALLY.
            for matrix_index in range(0, 2):

                for col in range(self.network_dimensions[matrix_index][1]):  # start in last column of matrix

                    for row in range(self.network_dimensions[matrix_index][0]):
                        self.network[matrix_index][row][col].learn()

            # STORE THE RESULTS IN THE PREDICTION LOG
            if len(self.prediction_log) > 1:
                new_epoch = self.prediction_log[len(self.prediction_log)-1][0] + 1
            else:
                new_epoch = 1
            learned_from = True
            Prediction_Start_Date = self.vector_date_list_training_set[v]._date_repr
            Input_Vector = input_vector_for_date
            Expected_Output_Vector = output_vector_for_date
            Actual_Output_Vector = output_list
            Errors = output_error_list

            self.prediction_log.append(
                [new_epoch, learned_from, Prediction_Start_Date, Input_Vector, Expected_Output_Vector,
                 Actual_Output_Vector, Errors])

            print(self.prediction_log[len(self.prediction_log)-1])

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

    def predict(self, input_vector, prediction_start_date):
        '''Predict the output(s) of the neural network, using the input_vector. Do not learn from / alter weights
        based on this input_vector.'''

        # CHARGE RECEPTORS WITH INPUT VECTOR
        for col in range(self.network_dimensions[0][1]):

            for row in range(self.network_dimensions[0][0]):
                self.network[0][row, col].set_charge(input_vector[row, col])  # set receptor's charge
                self.network = self.network[0][row, col].fire(self.network)  # fire and get network with changes

        # FEEDFORWARD INPUTS. *** FIRE BETWEEN NEURONS AND TO EFFECTORS
        for col in range(self.network_dimensions[1][1]):

            for row in range(self.network_dimensions[1][0]):
                self.network = self.network[1][row, col].fire(self.network)  # fire each neuron

        # CALCULATE ERRORS
        output_error_list = []
        output_list = []
        matrix_index = 2
        for col in reversed(range(self.network_dimensions[matrix_index][1])):

            for row in range(self.network_dimensions[matrix_index][0]):
                self.network[matrix_index][row, col].calculate_error()
                output_error_list.append(self.network[matrix_index][row, col].error)
                output_list.append(self.network[matrix_index][row, col].activated_charge)

        # STORE THE RESULTS IN THE PREDICTION LOG
        new_epoch = self.prediction_log[-1][0] + 1
        learned_from = False
        Prediction_Start_Date = prediction_start_date
        Input_Vector = input_vector
        Expected_Output_Vector = []
        Actual_Output_Vector = output_list
        Errors = output_error_list

        self.prediction_log.append(
            [new_epoch, learned_from, Prediction_Start_Date, Input_Vector, Expected_Output_Vector,
             Actual_Output_Vector, Errors])

    def predict_and_learn(self, input_vector):
        '''Take in the input_vector, predict an output and learn from that one input vector, by altering the
        weights in the network.'''

    def graph_error_over_time(self):
        '''Generate and return a graph from the error value(s) recorded for each input vector, along with the
        predicted and expected values.'''


class Receptor:
    '''The Receptor takes in an item from the input vector of the data, weights it and then fires the weighted value
    into the Neurons in the Neural Network. If the network is an ANN, then the weighted inputs are fired only into the
    first column of the neural matrix. If the network is an RC, then the weighted inputs are fired into each
    and every neuron in the neural matrix. A Receptor stores its location in the network.'''

    def __init__(self, charge=0, receptor_location=[0, 0], network_dimensions=[[1, 1], [1, 1], [1, 1]],
                 activation_function=STUNN.linear, learning_coefficient=0.05):
        '''Generate a random weight for itself. Weight the charge through multiplication and store value. Generate
        matrix of random weightings between -1 and 1, corresponding to the neuron matrix dimensions.'''

        self.charge = charge
        self.location = receptor_location
        self.network_dimensions = network_dimensions
        self.weight_matrix = self.create_weight_matrix()
        self.charged_weight_matrix = []
        self.activation_function = activation_function
        self.learning_coefficient = learning_coefficient

    def create_weight_matrix(self):
        '''Generate a matrix of weights the object will apply when firing to other objects in the network.'''

        neuron_matrix_row_count = self.network_dimensions[1][0]

        temp_matrix = np.random.rand(neuron_matrix_row_count, np.size([0]))

        return temp_matrix

    def set_charge(self, charge):
        '''Set the charge of the Receptor.'''
        self.charge = charge

    def fire(self, network):
        '''Fire the Receptor's charge into each of the neurons in the first column in the neuron matrix.'''
        self.processed_charge = self.activation_function(self.charge, deriv=False)

        self.charged_weight_matrix = self.processed_charge * np.array(
            self.weight_matrix)  # multiply charge into each weight
        neuron_matrix_row_count = self.network_dimensions[1][0]

        for neuron_index in range(neuron_matrix_row_count):
            network[1][neuron_index][0].add_charge(self.charged_weight_matrix[neuron_index][0])

        return network

    def reset_charge(self):
        '''Reset the charge of the Receptor back to zero.'''
        self.charge = 0

    def calculate_delta_error(self, network):
        '''Calculate the delta error value for this effector, using the delta errors of the first neuron layer.'''

        neuron_matrix_row_count = self.network_dimensions[1][0]

        weighted_delta_errors_of_following_layer = 0

        for row in range(neuron_matrix_row_count):
            weighted_delta_errors_of_following_layer += network[1][row][0].delta_error * \
                                                        self.weight_matrix[row][0]

        self.delta_error = weighted_delta_errors_of_following_layer * self.activation_function(
            self.activation_function(self.charge), deriv=True)

    def learn(self):
        '''Learn using the delta error value calculated for this object. Reset charge.'''
        for col in range(len(self.weight_matrix[0])):

            for row in range(len(self.weight_matrix)):
                self.weight_matrix[row][col] += self.delta_error * self.learning_coefficient * self.charge

        self.reset_charge()

        return True


class Neuron:
    '''The Neuron takes in charges from Receptors or from other Neurons, calculates the activation function value
    from the inputs and then applies each weight in its matrix of weights to these values from the activation function.
    The Neuron knows its location in the neural matrix it is apart of, which is contained in a network.'''

    def __init__(self, charge=0, neuron_location=[0, 0], network_dimensions=[[1, 1], [1, 1], [1, 1]],
                 activation_function=STUNN.sigmoid, learning_coefficient=0.05):
        '''Generate random weights for the neurons this neuron connects to and the effector(s) it connects to, if any.
        Adjust the weights as needed for the type of network. Store the weights in the neuron_network_weights variable.
        '''
        self.charge = 0
        self.location = neuron_location
        self.network_dimensions = network_dimensions
        self.weight_matrix = self.create_weight_matrix()
        self.charged_weight_matrix = []
        self.activation_function = activation_function  # save the activation function to be used by the neuron
        self.learning_coefficient = learning_coefficient

    def create_weight_matrix(self):
        '''Generate a matrix of weights the object will apply when firing to other objects in the network. If
        this is not the last column of the neuron matrix, an empty matrix the size of the existing neuron matrix
        will be generated and the following layer, relative to this neuron, will be filled with weights.'''

        neuron_matrix_row_count = self.network_dimensions[1][0]
        neuron_matrix_col_count = self.network_dimensions[1][1]
        neuron_col_location = self.location[1]
        effector_matrix_col_count = self.network_dimensions[2][1]  # number of columns in effector matrix
        effector_matrix_row_count = self.network_dimensions[2][0]

        # if neuron in last column of neuron matrix. create weight matrix to fire to effector matrix.
        if neuron_col_location == neuron_matrix_col_count - 1:
            temp_matrix = np.random.rand(effector_matrix_row_count, effector_matrix_col_count)
        else:  # create weight matrix to fire  to next layer of neuron matrix
            temp_matrix = np.random.rand(neuron_matrix_row_count, np.size([0]))

        return temp_matrix

    def add_charge(self, charge):
        '''Add charge to the neuron.'''
        self.charge += charge
        return self.charge

    def fire(self, network):
        '''Call methods to calculate activation function from charge and clear charge, apply potential to each weight
        and store in new matrix, and then fire weights into corresponding neurons and effectors.'''

        neuron_matrix_row_count = self.network_dimensions[1][0]
        neuron_matrix_col_count = self.network_dimensions[1][1]
        effector_matrix_row_count = self.network_dimensions[2][0]
        effector_matrix_col_count = self.network_dimensions[2][1]
        neuron_col_location = self.location[1]

        activated_charge = self.activation_function(self.charge, deriv=False)  # calculate neuron activation function

        self.charged_weight_matrix = activated_charge * np.array(self.weight_matrix)

        if neuron_col_location == neuron_matrix_col_count - 1:  # if firing to effectors, fire to all effectors

            for row in range(effector_matrix_row_count):

                for col in range(effector_matrix_col_count):
                    network[2][row][col].add_charge(self.charged_weight_matrix[row][col])

        else:  # if firing to next neuron layer, only fire to next neuron layer

            for row in range(neuron_matrix_row_count):
                network[1][row][neuron_col_location + 1].add_charge(self.charged_weight_matrix[row][0])

        return network

    def reset_charge(self):
        '''Reset charge of neuron back to zero.'''
        self.charge = 0
        return True

    def calculate_delta_error(self, network):
        '''Calculate the delta error value for this neuron, through the process of backpropagation. Assumes values have
        been calculated for all layers following.'''

        neuron_matrix_row_count = self.network_dimensions[1][0]
        neuron_matrix_col_count = self.network_dimensions[1][1]
        effector_matrix_row_count = self.network_dimensions[2][0]
        effector_matrix_col_count = self.network_dimensions[2][1]
        neuron_col_location = self.location[1]

        weighted_delta_errors_of_following_layer = 0

        if neuron_col_location == neuron_matrix_col_count - 1:  # if last neuron in neuron matrix

            for row in range(effector_matrix_row_count):

                for col in range(effector_matrix_col_count):
                    weighted_delta_errors_of_following_layer += network[2][row][col].delta_error * \
                                                                self.weight_matrix[row][col]

        else:

            for row in range(neuron_matrix_row_count):
                weighted_delta_errors_of_following_layer += \
                    network[1][row][neuron_col_location + 1].delta_error * \
                    self.weight_matrix[row][0]

        self.delta_error = weighted_delta_errors_of_following_layer * self.activation_function(
            self.activation_function(self.charge), deriv=True)

    def learn(self):
        '''Learn using the delta error value calculated for this object. Reset charge.'''

        for row in range(len(self.weight_matrix)):
            self.weight_matrix[row][0] += self.delta_error * self.learning_coefficient * self.charge

        self.reset_charge()

        return True


class Effector:
    '''The Effector is the end of the Network, where potentials fired from the neural matrix collect. The
    Effector stores the predicted output of the entire network. There may be multiple Effectors in a network.'''

    def __init__(self, charge=0, effector_location=[0, 0], expected_charge=0, activation_function=STUNN.linear,
                 learning_coefficient=0.05):
        '''Store potential value of zero. Store location of effector. Store expected potential.'''
        self.charge = charge
        self.location = effector_location
        self.expected_charge = expected_charge
        self.activation_function = activation_function
        self.learning_coefficient = learning_coefficient

    def add_charge(self, charge):
        '''Add potential to effector.'''
        self.charge = self.charge + charge
        return self.charge

    def reset_charge(self):
        '''Reset charge of effector to zero.'''
        self.charge = 0
        self.expected_charge = 0
        return True

    def set_expected_charge(self, expected_charge):
        '''Set the expected potential of the Effector'''
        self.expected_charge = expected_charge
        return self.expected_charge

    def calculate_error(self):
        '''Calculate and return the error between the potential of the effector and the expected potential.'''
        self.activated_charge = self.activation_function(self.charge)
        self.error = (self.activation_function(self.expected_charge) - self.activated_charge)

    def calculate_delta_error(self):
        '''Calculate the delta error for this effector. Reset charge.'''
        self.calculate_error()
        self.delta_error = self.error * self.activation_function(self.activated_charge, deriv=True)
        self.reset_charge()

    def get_info(self):
        '''Return information pertaining to the nature of this effector.'''
        return ['Location: ' + self.location, 'Charge: ' + self.charge, 'Expected Charge: ' + self.expected_charge,
                'Error: ' + self.error, 'Delta Error: ' + self.delta_error]


def list_dates_between_two_dates(start, end, delta):
    '''Find all the dates between two dates. Return the list of dates.'''
    curr = start
    while curr < end:
        yield curr
        curr += delta


if __name__ == '__main__':
    STUNN()