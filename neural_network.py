import math

class Activation:
    """Activation functions and their derivatives for backpropagation"""
    @staticmethod
    def linear():
        activation_func = lambda x : x
        d_activation_func = lambda: 1
        return (activation_func, d_activation_func)

    @staticmethod
    def tanh():
        activation_func = lambda x : (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
        d_activation_func = lambda x : 4 / (math.exp(x) + math.exp(-x))**2
        return (activation_func, d_activation_func)


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


     
