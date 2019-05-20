import math
import numpy as np


class Activation:
    """Activation functions and their derivatives for backpropagation"""
    @staticmethod
    def linear():
        activation_func = lambda x : x
        d_activation_func = lambda: 1
        return (activation_func, d_activation_func)

    @staticmethod
    def tanh():
        activation_func = lambda x : (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        d_activation_func = lambda x : 4 / (np.exp(x) + np.exp(-x))**2
        return (activation_func, d_activation_func)

class Error:
    @staticmethod
    def quadratic():
         error_func = lambda x : x**2
         d_error_func = lambda x : 2*x
         return (error_func, d_error_func)

#Matrix based approach
def create_MNIST_network():
    """
    Inputs are 28*28 and output is a 1*10 
    """
    layers = [784, 200, 50, 10]

class NeuralNetwork:

    def __init__(self,layer_dims,activation_func,diff_func,error_func):
        self.learning_rate = 0.01
        self.network = []
        #Zip to form pairs of dimensions corresponding to dims of weight matrices
        layers_weight_dimensions = list(zip(layer_dims[:-1],layer_dims[2:]))
        for weight_dim in layers_weight_dimensions:
            W_i = np.random.rand(weight_dim[0],weight_dim[1])
            self.network.append(W_i)
            

    def set_activation_func(self,activation_func, d_activation_func):
        self.activation_func = activation_func
        self.d_activation_func = d_activation_func
    
    def set_error_func(self,error_func, d_error_func):
        self.error_func = error_func
        self.d_error_func = d_error_func

    def train(self,input,output):
        """ Train network on a sample """
        layer_n = len(self.network)

        #Forward pass
        self.outputs_pre_activation = []
        self.outputs = []
        output_i = input
        #TODO probably incorrect intialization
        self.outputs_pre_activation.append(input)
        self.outputs.append(input)

        for layer_i in range(0,layer_n):
            #get next output 
            output_i = np.dot(self.network[layer_i],output_i)
            self.outputs_pre_activation.append(output_i)
            output_i = self.activation_func(output_i)
            self.outputs.append(output_i)

        #Backward propagation to compute derivatives
        computed_output = self.outputs[-1]
        error_derv = sum(2*(computed_output - output))
        self.network_d = []

        #initialize
        output_vec = self.outputs_pre_activation[-1]
        dE_dOi = error_derv*self.d_activation_func(output_vec)

        #Start from rightmost layer
        for layer_i in reversed(range(0,layer_n)):
            weight_mat = self.network[layer_i]
            output_vec = self.outputs_pre_activation[layer_i]
            input_vec = self.outputs[layer_i-1]

            #error derivative w.r.t to output vector before tanh
            #values before tanh needed

            #TODO what about earlier layers?
            
            dE_dWij = np.outer(dE_dOi, input_vec)
            
            #Weight updates !
            weight_mat += dE_dWij*self.learning_rate

            self.network[layer_i]
            
        #Gradient Descent 


    def update(self,input,desired_output):
        pass
    def test(self,input,output):
        """ Returns error of an input evaluated on this network """
        pass


#Old approach
class InputNeuron:
    """ Input Neuron that only outputs a constant value """
    def __init__(self, input):
        self.output = input

class Neuron:
    """Neuron object thats configurable with custom activation function"""
    def __init__(self,input_offset):
        """ 
        Parameters
        ----------
        input_offset : double
            Used to create intializer neurons, output is exactly
            equal to input_offset if there are no input connections
        """

        self.input_neurons = []
        self.output_neurons = []
        self.has_output = False

        self.set_activation_func(*Activation.tanh())
        self.diff_output = 0
        self.weight_update_constant = 0.05
        self.input_offset = input_offset

    def add_diff_output(self,output_val,output_derv):
        self.diff_output += self.diff_func(output_val)*output_derv
        #print (self.diff_output)

    def set_activation_func(self,activation_func,diff_func):
        self.activation_func = activation_func
        self.diff_func = diff_func
    
    def add_input(self,neuron, weight):
        self.input_neurons.append([neuron,weight])
    
    def adjust_weights(self):
        for idx in range(0,len(self.input_neurons)):
            child_neuron = self.input_neurons[idx][0]
            self.input_neurons[idx][1] *= (1+child_neuron.output*self.diff_output*0.1)
            #print (self.input_neurons[idx][1])

    def create_output(self):
        self.diff_output = 0
        input = 0
        input += self.input_offset
        for neuron,weight in self.input_neurons:
            if neuron.has_output:
                input += neuron.output*weight

        self.output = self.activation_func(input)
        self.has_output = True
        print ("Output is ",self.output)

if __name__ == "__main__":
    target = 0.8
    print ("Target is 0.7")
    neuron_l1_1 = Neuron(0.5)
    neuron_l1_2 = Neuron(1.0)
    neuron_l2_1 = Neuron(0.5)
    neuron_l2_1.add_input(neuron_l1_1, 0.5)
    neuron_l2_1.add_input(neuron_l1_2, 0.5)
    
    for x in range(0,100):
        #forward passp- 
        neuron_l1_1.create_output()
        neuron_l1_2.create_output()
        neuron_l2_1.create_output()

        error_val = (target- neuron_l2_1.output)**2
        print ("error is ", error_val)
        error_l2_1_diff = (1.0)*2*(target - neuron_l2_1.output)#de/do2_1
        #print (error_l2_1_diff)
        #error propogation
        neuron_l2_1.add_diff_output(error_val, error_l2_1_diff)
        #print (neuron_l2_1.diff_output)
        neuron_l1_1.add_diff_output(neuron_l2_1.output,neuron_l2_1.diff_output)
        neuron_l1_2.add_diff_output(neuron_l2_1.output,neuron_l2_1.diff_output)

        #update
        neuron_l2_1.adjust_weights()
        neuron_l1_1.adjust_weights()
        neuron_l1_2.adjust_weights()


     
