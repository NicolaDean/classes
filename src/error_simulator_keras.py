import tensorflow as tf
import numpy as np
from .injection_sites_generator import InjectableSite, InjectionSitesGenerator
from enum import IntEnum
import sys

def create_injection_sites_layer_simulator(num_requested_injection_sites, layer_type, layer_output_shape_cf,
                                           layer_output_shape_cl, models_folder, return_id_errors = False):
    def __generate_injection_sites(sites_count, layer_type, size, models_mode=''):

        injection_site = InjectableSite(layer_type, '', size)
        try:
            injection_sites, cardinality, pattern = InjectionSitesGenerator([injection_site],
                                                                            models_mode, models_folder)\
                .generate_random_injection_sites(sites_count)
        except:
            return []
        return injection_sites, cardinality, pattern

    available_injection_sites = []
    masks = []

    ids = []        #list of error models

    for _ in range(num_requested_injection_sites):
        curr_injection_sites, cardinality, pattern = __generate_injection_sites(1, layer_type, layer_output_shape_cf)
        shape = eval(layer_output_shape_cl.replace('None', '1'))
        curr_inj_nump = np.zeros(shape=shape[1:])
        curr_mask = np.ones(shape=shape[1:])

        if len(curr_injection_sites) > 0:
            ids.append((cardinality, pattern))
            for idx, value in curr_injection_sites[0].get_indexes_values():
                channel_last_idx = (idx[0], idx[2], idx[3], idx[1])
                if value.value_type == '[-1,1]':
                    curr_inj_nump[channel_last_idx[1:]] += value.raw_value
                else:
                    curr_mask[channel_last_idx[1:]] = 0
                    curr_inj_nump[channel_last_idx[1:]] += value.raw_value

            available_injection_sites.append(curr_inj_nump)
            masks.append(curr_mask)

    if return_id_errors:
        return available_injection_sites, masks, ids
    else:
        return available_injection_sites, masks

class ErrorSimulatorMode(IntEnum):
    disabled = 1,
    enabled  = 2

'''
Inject Different fault for each image inside a batch
'''
@tf.function
def fault_injection_batch(inputs,__num_inj_sites,__available_injection_sites,__masks):
    #Extract batch Size
    shape       = tf.shape(inputs)
    batch_size  = shape[0]

    #Compute fault on first image
    init = fault_injection(inputs[0],__num_inj_sites,__available_injection_sites,__masks)
    i=tf.constant(1)

    '''
    While exit condition
    '''
    def condition(i,_):
        return i<batch_size
    
    '''
    While body functiom, execute fault injection on each element of the batch
    '''
    def iteration(i,outputs):
        tmp = fault_injection(inputs[i],__num_inj_sites,__available_injection_sites,__masks)
        outputs = tf.concat([outputs,tmp],0)
        i+=1
        return [i,outputs]
    
    #Execute a while loop of batch size to apply fault on full batch
    i, outputs = tf.while_loop(condition, iteration,
                                [i, init],
                                [i.get_shape(), tf.TensorShape([None,init.get_shape()[1],init.get_shape()[2],init.get_shape()[3]])])
    
    #tf.print(outputs)
    return outputs
    

    '''
    Inject a Fault from the generated injection sites on the selected input
    '''
def fault_injection(inputs,__num_inj_sites,__available_injection_sites,__masks):
    random_index = tf.random.uniform(
        shape=[1], minval=0,
        maxval=__num_inj_sites, dtype=tf.int32, seed=22)

    #tf.print(random_index.shape)
    #print(f"Fault from {self.name}")
    random_tensor   = tf.gather(__available_injection_sites, random_index)
    random_mask     = tf.gather(__masks, random_index)

    #tf.print(random_tensor.shape,random_mask.shape)
    #return [inputs[i] * random_mask + random_tensor, random_tensor, random_mask]
    return inputs * random_mask + random_tensor

@tf.function
def fault_injection_batch_v2(inputs,__num_inj_sites,__available_injection_sites,__masks, error_simulator):
    shape       = tf.shape(inputs)
    batch_size  = shape[0]

    random_indexes = tf.random.uniform(
        shape=[batch_size], minval=0,
        maxval=__num_inj_sites, dtype=tf.int32, seed=22)    
    
    random_tensor   = tf.gather(__available_injection_sites, random_indexes)
    random_mask     = tf.gather(__masks, random_indexes)
    
    random_indexes = tf.expand_dims(random_indexes, axis = 1)

    error_simulator.history.assign(tf.concat([error_simulator.history, random_indexes], axis = 0))
    #tf.print(random_tensor.shape,random_mask.shape)
    return inputs * random_mask + random_tensor

class ErrorSimulator(tf.keras.layers.Layer):

    def __init__(self, available_injection_sites, masks, num_inj_sites, error_ids, **kwargs):

        super(ErrorSimulator, self).__init__(**kwargs)
        self.__num_inj_sites = num_inj_sites
        self.__available_injection_sites = []
        self.__masks = []
        self.__cardinalities = []
        self.error_ids = error_ids
        self.history = tf.Variable(initial_value = [[1]], dtype = tf.int32, shape = [None, 1], trainable=False)
        #Parameter to chose between enable/disable faults
        self.mode = tf.Variable([[int(ErrorSimulatorMode.enabled)]],shape=tf.TensorShape((1,1)),trainable=False) 
        
        for inj_site in available_injection_sites:
            self.__available_injection_sites.append(tf.convert_to_tensor(inj_site, dtype=tf.float32))
        for mask in masks:
            self.__masks.append(tf.convert_to_tensor(mask, dtype=tf.float32))

    '''
    Allow to enable or disable the Fault Layer
    '''
    def set_mode(self, mode:ErrorSimulatorMode):
        self.mode.assign([[int(mode)]])

    def get_history(self):
        return self.history.numpy()
    
    def clear_history(self):
        self.history.assign(tf.Variable(initial_value = [[1]], dtype = tf.int32, shape = [None, 1], trainable=False))
        
    
    #TODO MOVE THE GENERATION OF INDEXES OUTSIDE THE WHILE LOOP IN A SINGLE INSTRUCTION (faster)
    @tf.custom_gradient
    def call(self, inputs):
        #tf.print("MODE LAYER :", self.mode, tf.constant([[int(ErrorSimulatorMode.disabled)]]), output_stream=sys.stdout)
        #TF operator to check which mode is active
        #If Disabled => Return Vanilla output
        #If Enabled  => Return Faulty  output
        
        def grad(upstream):
            return upstream
        
        
    
        return tf.cond(tf.math.logical_or(self.mode == tf.constant([[int(ErrorSimulatorMode.disabled)]]),
                                           tf.random.uniform(shape = [], maxval = 100, dtype = tf.int32) < 0), 
                       true_fn=lambda: inputs,
                       false_fn=lambda: fault_injection_batch_v2(inputs,self.__num_inj_sites,self.__available_injection_sites,self.__masks, self)), grad
    
    '''
    #OLD DEPRECATED CALL FUNCTION
    def call(self, inputs):
        random_index = tf.random.uniform(
            shape=[1], minval=0,
            maxval=self.__num_inj_sites, dtype=tf.int32, seed=22)

        random_tensor = tf.gather(self.__available_injection_sites, random_index)
        random_mask = tf.gather(self.__masks, random_index)
        return [inputs * random_mask + random_tensor, random_tensor, random_mask]
    '''

    '''
    def fault_injection(self,inputs,a,b,c):
        random_index = tf.random.uniform(
            shape=[1], minval=0,
            maxval=self.__num_inj_sites, dtype=tf.int32, seed=22)

        random_tensor = tf.gather(self.__available_injection_sites, random_index)
        random_mask = tf.gather(self.__masks, random_index)
        output = inputs * random_mask + random_tensor

        #tf.print(output)
        return output
    '''



    '''

    random_index = tf.random.uniform(
            shape=[batch_size,1], minval=0,
            maxval=self.__num_inj_sites, dtype=tf.int32, seed=22)

    masks       = tf.broadcast_to(masks,(indices[0],masks[1],...,))
    injections  = tf.broadcast_to(self.__available_injection_sites,(indices[0],injections[1],...,))

    masks       = tf.gather(mask,indices,batch_dims=1)
    injections  = tf.gather(injections,indices,batch_dims=1)

    return inputs*masks + injections


    un altra soluzione è usare tf.where(masks,inputs,injections)
    '''