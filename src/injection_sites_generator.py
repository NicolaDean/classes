import os
import json
from collections import OrderedDict
import numpy as np
import tensorflow as tf
import time
import enum
import operator
import functools
import traceback
import struct
import os.path
from sys import exit


class OperatorType(enum.Enum):
    """
    The TensorFlow's injectables operators.
    This class stores the type (name of the operator) and allows
    to convert the operator type from TensorFlow's name to CAFFE model's name.
    """
    Conv2D1x1 = 1  # Convolution 2D with kernel size of 1.
    Conv2D3x3 = 2  # Convolution 2D with kernel size of 3.
    Conv2D3x3S2 = 3  # Convolution 2D with kernel size of 3 and stride of 2.
    AddV2 = 4  # Add between two tensors.
    BiasAdd = 5  # Add between a tensor and a vector.
    Mul = 6  # Mul between a tensor and a scalar.
    FusedBatchNormV3 = 7  # Batch normalization.
    RealDiv = 8  # Division between a tensor and a scalar.
    Exp = 9  # Exp activation function.
    LeakyRelu = 10  # Leaky Relu activation function.
    Sigmoid = 11  # Sigmoid activation function.
    Add = 12  # Add between two tensors.
    Conv2D = 13
    FusedBatchNorm = 14

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def get_model_name(self):
        """
        Returns the CAFFE model's name from the TensorFlow's operator type.

        Returns:
            [str] -- The CAFFE model's name.
        """
        # Switch statement that map each TF's operator type to the model names
        # used in the simulations.
        if self == OperatorType.Conv2D1x1:
            return "S1_convolution"
        elif self == OperatorType.Conv2D3x3:
            return "S2_convolution"
        elif self == OperatorType.Conv2D3x3S2:
            return "S3_convolution"
        elif self == OperatorType.AddV2:
            return "S1_add"
        elif self == OperatorType.BiasAdd:
            return "S1_biasadd"
        elif self == OperatorType.Mul:
            return "S1_mul"
        elif self == OperatorType.FusedBatchNormV3:
            return "S1_batch_norm"
        elif self == OperatorType.Exp:
            return "S1_exp"
        elif self == OperatorType.LeakyRelu:
            return "S1_leaky_relu"
        elif self == OperatorType.Sigmoid:
            return "S1_sigmoid"
        elif self == OperatorType.RealDiv:
            return "S1_div"
        elif self == OperatorType.Add:
            return "S2_add"
        elif self == OperatorType.Conv2D:
            return "S1_convolution"
        elif self == OperatorType.FusedBatchNorm:
            return "S1_batch_norm"
        else:
            raise ValueError("Unable to find a model for this operator: {}".format(self))

    @staticmethod
    def all():
        """
        Returns all the types as list.

        Returns:
            [list] -- List of operator types.
        """
        return list(OperatorType)

    @staticmethod
    def all_aliases():
        """
        Returns the model's names associated to each operator type.

        Returns:
            [list] -- List of model's names
        """
        return [operator.get_model_name() for operator in OperatorType.all()]


class InjectableSite(object):
    """
    Describes an injectable operator and it is characterized by the TensorFlow's operator type,
    the TensorFlow's operator graph name and the size of the output tensor.
    """

    def __init__(self, operator_type, operator_name, size):
        """
        Creates the object with the operator type, name and size.

        Arguments:
            operator_type {OperatorType} -- TensorFlow's operator type.
            operator_name {str} -- TensorFlow's operator graph name.
            size {str} -- The output tensor size expressed as string.
        """
        self.__operator_type = operator_type
        self.__operator_name = operator_name
        size = size.replace("None", "1")
        tuple_size = eval(size)

        # If the size has less than 4 components, it is expanded to match a tensor shape.
        if len(tuple_size) == 4:
            self.__has_all_components = True
            self.__size = size
            self.__components = 0
        else:
            remainder = 4 - len(tuple_size)
            self.__has_all_components = False
            self.__components = remainder
            self.__size = str(tuple([1] * remainder + list(tuple_size)))

    def __repr__(self):
        return "InjectableSite[Type: {}, Name: {}, Size: {}]".format(self.__operator_type, self.__operator_name,
                                                                     self.__size)

    def __str__(self):
        return self.__repr__()

    def __get__size(self):
        if self.__has_all_components:
            return self.__size
        else:
            size_eval = eval(self.__size)
            size = [size_eval[i] for i in range(self.__components, len(size_eval))]
            return str(tuple(size))

    operator_type = property(fget=lambda self: self.__operator_type)
    operator_name = property(fget=lambda self: self.__operator_name)
    size = property(fget=__get__size)


class InjectionValue(object):
    """
    Represents a value to be injected in an operator's output tensor.
    There can be 4 types of values:
    NaN, inserts a NaN value, Zeroes, inserts a zero value,
    [-1, 1], inserts a difference between -1 and 1 (zero excluded) and
    Others, which represents a random 32-wide bitstring.
    """

    @staticmethod
    def nan():
        return InjectionValue("NaN", np.float32(np.nan))

    @staticmethod
    def zeroes():
        return InjectionValue("Zeroes", np.float32(0.0))

    @staticmethod
    def between_one(raw_value):
        return InjectionValue("[-1,1]", np.float32(raw_value))

    @staticmethod
    def others(raw_value):
        return InjectionValue("Others", np.float32(raw_value))

    def __init__(self, value_type, raw_value):
        self.__value_type = value_type
        self.__raw_value = raw_value

    def __str__(self):
        return "({}, {})".format(self.__value_type, hex(struct.unpack('<I', struct.pack('<f', self.__raw_value))[0]))

    value_type = property(fget=lambda self: self.__value_type)
    raw_value = property(fget=lambda self: self.__raw_value)


class InjectionSite(object):
    """
    Represents an injection site and is composed by the operator name to inject,
    the indexes where insert the injections and the values to insert.

    It can be iterated to get pairs of indexes and values.
    """

    def __init__(self, operator_name):
        self.__operator_name = operator_name
        self.__indexes = []
        self.__values = []

    def add_injection(self, index, value):
        self.__indexes.append(index)
        self.__values.append(value)

    def __iter__(self):
        self.__iterator = zip(self.__indexes, self.__values)
        return self

    def next(self):
        next_element = next(self.__iterator)
        if next_element is None:
            raise StopIteration
        else:
            return next_element

    def get_indexes_values(self):
        return zip(self.__indexes, self.__values)

    operator_name = property(fget=lambda self: self.__operator_name)

    def to_json(self):
        json_representation = {}
        for index, value in self:
            json_representation[str(index)] = str(value)
        json_representation["operator_name"] = self.__operator_name
        return json_representation


operator_names_table = {
    "S1_add": "AddV2",
    "S2_add": "Add",
    "S1_batch_norm": "FusedBatchNormV3",
    "S1_biasadd": "BiasAdd",
    "S1_convolution": "Conv2D1x1",
    "S1_div": "RealDiv",
    "S1_exp": "Exp",
    "S1_leaky_relu": "LeakyRelu",
    "S1_mul": "Mul",
    "S1_sigmoid": "Sigmoid",
    "S2_convolution": "Conv2D3x3",
    "S3_convolution": "Conv2D3x3S2",
    "S1_convolution_test": "Conv2D1x1"
    }

NAN = "NaN"
ZEROES = "Zeroes"
BETWEEN_ONE = "[-1, 1]"
OTHERS = "Others"

SAME_FEATURE_MAP_SAME_ROW = 0
SAME_FEATURE_MAP_SAME_COLUMN = 1
SAME_FEATURE_MAP_BLOCK = 2
SAME_FEATURE_MAP_RANDOM = 3
MULTIPLE_FEATURE_MAPS_BULLET_WAKE = 4
MULTIPLE_FEATURE_MAPS_BLOCK = 5
MULTIPLE_FEATURE_MAPS_SHATTER_GLASS = 6
MULTIPLE_FEATURE_MAPS_QUASI_SHATTER_GLASS = 7
MULTIPLE_FEATURE_MAPS_UNCATEGORIZED = 8

BLOCK_SIZE = 16


class InjectionSitesGenerator(object):
    def __init__(self, injectable_sites, mode):
        self.__injectable_sites = injectable_sites
        self.__cardinalities = self.__load_cardinalities()
        self.__corrupted_values_domain = self.__load_corrupted_values_domain(mode)
        self.__spatial_models = self.__load_spatial_models()
        self.__debugcardinality = -1

    def generate_random_injection_sites(self, size):

        injectables_site_indexes = np.random.choice(len(self.__injectable_sites), size=size)
        injection_sites = []
        cardinalities = []
        patterns = []
        for index in injectables_site_indexes:
            run = True
            while run:
                try:
                    injectable_site = self.__injectable_sites[index]
                    operator_type = injectable_site.operator_type.name
                    if operator_type == 'Conv2D':
                        operator_type = 'Conv2D1x1'
                    if operator_type == 'FusedBatchNorm':
                        operator_type = 'FusedBatchNormV3'
                    cardinality = self.__select_cardinality(self.__cardinalities[operator_type])
                    self.__debugcardinality = cardinality
                    injection_site = InjectionSite(injectable_site.operator_name)
                    corrupted_values = self.__select_multiple_corrupted_values(
                        self.__corrupted_values_domain[operator_type], cardinality)
                    indexes = self.__select_spatial_pattern(self.__spatial_models[operator_type], cardinality,
                                                            eval(injectable_site.size))
                    for idx, value in zip(indexes, corrupted_values):
                        injection_site.add_injection(idx, value)
                    injection_sites.append(injection_site)
                    cardinalities.append(self.__debugcardinality)
                    patterns.append(self.__debugspatial_model)
                    run = False
                except Exception as exception:
                    print(exception)
                    traceback.print_exc()
                    continue

        def dumper(o):
            try:
                return o.to_json()
            except:
                return o.__dict__

        #with open("injection_sites.json", "w") as injection_sites_json_file:
        #    json.dump(injection_sites, injection_sites_json_file, default=dumper)
        return injection_sites, cardinalities, patterns

    def __get_models(self):
        models = set()
        for injectable_site in self.__injectable_sites:
            if injectable_site.operator_type not in models:
                models.add(injectable_site.operator_type)
        temp_names = [model.get_model_name() for model in models]
        return temp_names

    def __load_cardinalities(self):
        """
        Loads the cardinalities for each model.
        It creates a dictionary for each model, containing
        the cardinalities and their probability distribution.

        Returns:
            [dict] -- Returns a dictionary, having the models as keys and the
            cardinalities, associated to each model, as values.
        """
        cardinalities = {}  # Map of cardinalities for each model.
        for model_operator_name in self.__get_models():  # operator_names_table.keys():
            # print("current model operator name ", model_operator_name)
            # Folders are named as "SX_model", while files are names "model_SX"
            # So, it is needed to reverse the model name to compose the cardinalities file path,
            separator = model_operator_name.index("_")
            model_prefix = model_operator_name[:separator], model_operator_name[separator + 1:]
            experiment_name = model_prefix[1] + "_" + model_prefix[0]
            base_path = os.path.dirname(os.path.realpath(__file__))
            model_cardinalities_path = base_path + "/models/{}/{}_anomalies_count.json".format(model_operator_name,
                                                                                               experiment_name)
            # Open the cardinalities file path and load it as a json file.
            with open(model_cardinalities_path, "r") as cardinalities_json:
                model_cardinalities = json.load(cardinalities_json)

                # print(model_cardinalities)

                # Add each cardinalities model to the map.
                # The insertion is done in order, so keys (the cardinalities) are sorted
                # and converted from string (json) to integer.
                # Only the probability of each cardinality is preserved, the absolute frequence
                # is not relevent for this problem.
                if operator_names_table[model_operator_name] not in cardinalities:
                    cardinalities[operator_names_table[model_operator_name]] = OrderedDict()

                # print(cardinalities)

                for cardinality in sorted(model_cardinalities.keys(), key=lambda x: int(x)):
                    cardinalities[operator_names_table[model_operator_name]][int(cardinality)] = float(
                        model_cardinalities[cardinality][1])

                probabilities_left = 1.0 - sum(cardinalities[operator_names_table[model_operator_name]].values())
                cardinalities[operator_names_table[model_operator_name]][int(cardinality)] += probabilities_left
        return cardinalities

    def __load_corrupted_values_domain(self, mode):
        """
        Loads the corrupted values domain for each model.
        It creates a dictionary for each model, containing
        the domains and their probability distribution.

        Returns:
            [dict] -- Returns a dictionary having the models as keys and the
            corrupted values domains, associated to each model, as values.
        """

        def extract_value(line):
            # Each line is composed by an identifier, colon and then the float value.
            # "Identifier: 0.345345"
            # The line is split according the colon, and is selected the component containing
            # the float value, avoiding the last character that is "\n".
            # Then is simply converted to float.
            try:
                value = float(line.split(":")[1][1:])
            except IndexError:
                value = 0.0
            return value

        def get_line(lines, word):
            # Searches the first line, if present, that contains the word parameter.
            # Otherwise, returns an empty string.
            for line in lines:
                if word in line:
                    return line
            return ""

        corrupted_values_domain = {}  # Map of models and their corrupted values domain.
        for model_operator_name in self.__get_models():
            # The file is simply named as "value_analysis" and is common to each model.
            base_path = os.path.dirname(os.path.realpath(__file__))
            value_path = base_path + "/models/{}/value_analysis.txt".format(model_operator_name)
            with open(value_path, "r") as value_analysis_file:
                # Read the files as text lines.
                model_corrupted_values_domain = OrderedDict()
                lines = value_analysis_file.readlines()
                # Extracts the NaN, Zeroes, [-1, 1] and Others probabilities.
                model_corrupted_values_domain[NAN] = extract_value(get_line(lines, "NaN"))
                model_corrupted_values_domain[ZEROES] = extract_value(get_line(lines, "Zeros"))
                valid_scale_factor = extract_value(get_line(lines, "Valid"))
                model_corrupted_values_domain[BETWEEN_ONE] = extract_value(
                    get_line(lines, "[-1, 1]")) * valid_scale_factor
                model_corrupted_values_domain[OTHERS] = extract_value(get_line(lines, "Others")) * valid_scale_factor
                probability_left = 1.0 - sum(model_corrupted_values_domain.values())
                model_corrupted_values_domain[OTHERS] += probability_left
                # Set the corrupted domain to the relative model.
                corrupted_values_domain[operator_names_table[model_operator_name]] = model_corrupted_values_domain
        return corrupted_values_domain

    def __load_spatial_models(self):
        spatial_models = {}
        for model_operator_name in self.__get_models():
            base_path = os.path.dirname(os.path.realpath(__file__))
            spatial_model_path = base_path + "/models/{}/{}_spatial_model.json".format(model_operator_name,
                                                                                       model_operator_name)
            with open(spatial_model_path, "r") as spatial_model_json:
                spatial_model = json.load(spatial_model_json)
                if operator_names_table[model_operator_name] not in spatial_models:
                    spatial_models[operator_names_table[model_operator_name]] = spatial_model
        return spatial_models

    def __unpack_table(self, table):
        """
        Given a lookup table, implemented as a dictionary, it separates the keys from values
        and returns them in pairs but in different lists.
        Arguments:
            table {dict} -- Lookup table.

        Returns:
            [list, list] -- Returns two lists, the first one contains the keys while the
            latter contains the values.
        """
        keys = []
        values = []
        # Move each pair of key and value to two separate
        # lists to preverse the order.
        for key, value in table.items():
            keys.append(key)
            values.append(value)
        return keys, values

    def __random(self, options, probabilities, samples=1):
        """
        Selects one or multiple option(s) according to the probability
        distribution given.

        Arguments:
            options {list} -- List of options.
            probabilities {list(float)} -- List of floats that describes the probability
            distribution associated to options.

        Keyword Arguments:
            samples {int} -- Number of samples to selects. (default: {1})

        Returns:
            [scalar or list] -- Returns a scalar option if samples is 1, otherwise
            returns a list of options.
        """
        # Return a random option, or more than one, according to the probabilities distribution.
        # The if is needed because specifying size set to 1 returns an array instead of a scalar.
        # In case of samples > 1, is the intended behavior.
        for i in range(len(probabilities)):
            if probabilities[i] < 0.0:
                probabilities[i] = 0.0
        if sum(probabilities) < 1.0:
            remainder = 1.0 - sum(probabilities)
            probabilities[-1] += remainder
        if samples > 1:
            # z = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(probabilities), 0, 1)))
            # values, indices = tf.math.top_k(tf.math.log(probabilities) + z, samples)
            # return tf.gather(options, indices).to_list()
            return np.random.choice(options, size=samples, p=probabilities).tolist()
        else:
            # z = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(probabilities), 0, 1)))
            # values, indices = tf.math.top_k(tf.math.log(probabilities) + z, samples)
            # return tf.gather(options, indices)
            return np.random.choice(options, p=probabilities)

    def __select_cardinality(self, model_cardinalities):
        """
        Selects a cardinality among the ones provided for the model according to
        the probability distribution associated.

        Arguments:
            model_cardinalities {dict} -- Dictionary containing the integer cardinalities
            and the probabilities as keys.

        Returns:
            {int} -- Returns the drawn cardinality.
        """
        return self.__random(*self.__unpack_table(model_cardinalities))

    def __select_corrupted_value(self, model_corrupted_values_domain):
        """
        Selects a domain among the availables (NaN, Zeroes, [-1, 1] and Others) and
        a value from that domain.

        For NaN, Zeroes and Others the value returned is ready to use or to be inserted,
        while for [-1, 1] the value has to be added to the target value.

        Arguments:
            model_corrupted_values_domain {dict} -- Dictionary containing the domains (strings) as keys and
            their probabilities as values.

        Returns:
            [tuple(string, numpy.float32)] -- Returns a tuple that contains the domain and
            a value from that domain.
        """
        # Selects a domain among the NaN, Zeroes, [1, -1] and Others.
        domain = self.__random(*self.__unpack_table(model_corrupted_values_domain))
        if domain == NAN:
            # Returns a F32 NaN.
            return InjectionValue.nan()
        elif domain == ZEROES:
            # Returns a F32 zero.
            return InjectionValue.zeroes()
        elif domain == BETWEEN_ONE:
            # Returns a F32 between -1 and 1.
            return InjectionValue.between_one(np.random.uniform(low=-1.0, high=1.001))
        elif domain == OTHERS:
            # Returns a 32-long bitstring, interpreted as F32.
            bitstring = "".join(np.random.choice(["0", "1"], size=32))
            integer_bitstring = int(bitstring, base=2)
            float_bitstring = np.frombuffer(np.array(integer_bitstring), dtype=np.float32)[0]
            return InjectionValue.others(float_bitstring)

    def __select_multiple_corrupted_values(self, model_corrupted_values_domain, size):
        """
        Returns multiple pairs of (domain, value) as many as the indicated size.
        It behaves like the similar scalar method.

        Arguments:
            model_corrupted_values_domain {dict} -- Dictionary containing the domains (strings) as keys and
            their probabilities as values.
            size {int} -- Number of tuple to generate.

        Returns:
            [list(tuple(string, numpy.float32))] -- Returns a list of tuple, containing the domain and
            a value from that domain.
        """
        return [self.__select_corrupted_value(model_corrupted_values_domain) for _ in range(size)]

    def __select_spatial_pattern(self, spatial_model, cardinality, output_size):
        def multiply_reduce(iterable):
            """
            Given an iterable multiplieas each element
            :param iterable:
            :return: multiplication of each element
            """
            return functools.reduce(operator.mul, iterable, 1)

        def random_same_feature_map(output_size, max_offset, scale_factor, cardinality):
            random_feature_map = np.random.randint(low=0, high=output_size[1])
            feature_map_size = output_size[2] * output_size[3]
            if max_offset * scale_factor >= feature_map_size:
                max_offset = int(feature_map_size / scale_factor) - 1
            random_starting_index = np.random.randint(low=0, high=feature_map_size - max_offset * scale_factor)
            random_starting_index += random_feature_map * feature_map_size
            offsets = np.random.choice(max_offset, replace=False, size=cardinality - 1)
            indexes = [random_starting_index]
            for offset in offsets:
                indexes.append(random_starting_index + offset * scale_factor)
            return [np.unravel_index(index, shape=output_size) for index in indexes]

        def random_pattern(fault_type, output_size, pattern, cardinality):
            if fault_type == SAME_FEATURE_MAP_SAME_ROW:
                return random_same_feature_map(output_size, int(patterns["MAX"]), 1, cardinality)
            elif fault_type == SAME_FEATURE_MAP_SAME_COLUMN:
                return random_same_feature_map(output_size, int(patterns["MAX"]), output_size[3], cardinality)
            elif fault_type == SAME_FEATURE_MAP_BLOCK:
                return random_same_feature_map(output_size, int(patterns["MAX"]), 16, cardinality)
            elif fault_type == SAME_FEATURE_MAP_RANDOM:
                random_feature_map = np.random.randint(low=0, high=output_size[1])
                feature_map_size = output_size[2] * output_size[3]
                indexes = np.random.choice(feature_map_size, size=cardinality, replace=False)
                return [
                    np.unravel_index(index + random_feature_map * feature_map_size, shape=output_size)
                    for index in indexes
                    ]
            elif fault_type == MULTIPLE_FEATURE_MAPS_BULLET_WAKE:
                max_feature_map_offset = int(patterns["MAX"])
                if max_feature_map_offset >= output_size[1]:
                    max_feature_map_offset = output_size[1] - 1
                feature_map_index = np.random.randint(low=0, high=output_size[1] - max_feature_map_offset)
                try:
                    feature_map_offsets = np.random.choice(max_feature_map_offset, size=cardinality - 1, replace=False)
                except ValueError:
                    feature_map_offsets = np.random.choice(max_feature_map_offset, size=cardinality - 1, replace=True)
                feature_map_indexes = [feature_map_index]
                for offset in feature_map_offsets:
                    feature_map_indexes.append(feature_map_index + offset)
                feature_map_size = output_size[2] * output_size[3]
                random_index = np.random.randint(low=0, high=feature_map_size)
                return [
                    np.unravel_index(random_index + feature_map_index * feature_map_size, shape=output_size)
                    for feature_map_index in feature_map_indexes
                    ]
            elif fault_type == MULTIPLE_FEATURE_MAPS_BLOCK:
                max_block_offset = int(patterns["MAX"])
                if max_block_offset * 16 >= max_linear_index:
                    max_block_offset = int(max_linear_index / 16) - 1
                starting_index = np.random.randint(low=0, high=max_linear_index - max_block_offset * 16)
                offsets = np.random.choice(max_block_offset, replace=False, size=cardinality - 1)
                indexes = [starting_index]
                for offset in offsets:
                    indexes.append(starting_index + offset * 16)
                return [
                    np.unravel_index(index, shape=output_size)
                    for index in indexes
                    ]
            elif fault_type == MULTIPLE_FEATURE_MAPS_SHATTER_GLASS:
                max_offsets = [
                    int(patterns["MAX"][0]),
                    int(patterns["MAX"][1]),
                    int(patterns["MAX"][2]),
                    int(patterns["MAX"][3])
                    ]
                try:
                    feature_map_indexes = np.random.choice(output_size[1], replace=False, size=max_offsets[0])
                except:
                    feature_map_indexes = np.random.choice(output_size[1], replace=True, size=max_offsets[0])
                common_index = np.random.randint(low=0, high=output_size[2] * output_size[3])
                random_feature_map = np.random.choice(feature_map_indexes)
                remainder = cardinality - len(feature_map_indexes)
               # print("offsets max_offsets[2] {}, max_offsets[3] {}".format(max_offsets[2], max_offsets[3]))
                choices = list(range(max_offsets[2], max_offsets[3]))
                choices.remove(0)
                offsets = np.random.choice(choices, size=remainder)
                indexes = []
                for feature_map_index in feature_map_indexes:
                    indexes.append(feature_map_index * output_size[2] * output_size[3])
                for offset in offsets:
                    indexes.append(random_feature_map * output_size[2] * output_size[3] + offset)
                indexes = [idx for idx in indexes if idx >= 0]
                return [
                    np.unravel_index(index, shape=output_size)
                    for index in indexes
                    ]
            elif fault_type == MULTIPLE_FEATURE_MAPS_QUASI_SHATTER_GLASS:
                max_offsets = int(patterns["MAX"])
                feature_map_indexes = np.random.choice(output_size[1], replace=False, size=max_offsets)
                common_index = np.random.randint(low=0, high=output_size[2] * output_size[3])
                random_feature_map = np.random.choice(feature_map_indexes)
                remainder = cardinality - len(feature_map_indexes)
                choices = range(max_offsets[2], max_offsets[3])
                choices.remove(0)
                offsets = np.random.choice(choices, size=remainder)
                indexes = []
                for feature_map_index in feature_map_indexes:
                    if feature_map_index != random_feature_map:
                        indexes.append(feature_map_index * output_size[2] * output_size[3])
                for offset in offsets:
                    indexes.append(random_feature_map * output_size[2] * output_size[3] + offset)
                return [
                    np.unravel_index(index, shape=output_size)
                    for index in indexes
                    ]
            else:
                indexes = np.random.choice(max_linear_index, size=cardinality, replace=False)
                return [
                    np.unravel_index(index, shape=output_size)
                    for index in indexes
                    ]

        max_linear_index = multiply_reduce(output_size)
        if cardinality == 1:
            self.__debugspatial_model = -1
            selected_index = np.random.randint(low=0, high=max_linear_index)
            return [np.unravel_index(selected_index, shape=output_size)]
        else:
            fault_type = self.__random(*self.__unpack_table(spatial_model[str(cardinality)]["FF"]))
            self.__debugspatial_model = int(fault_type)
            patterns = spatial_model[str(cardinality)]["PF"][fault_type]
            fault_type = int(fault_type)
            if len(patterns) == 2 and "MAX" in patterns and "RANDOM" in patterns:
                return random_pattern(fault_type, output_size, patterns, cardinality)
            revised_patterns = patterns.copy()
            revised_patterns.pop("MAX", None)
            pattern = self.__random(*self.__unpack_table(revised_patterns))
            if pattern == "RANDOM":
                return random_pattern(fault_type, output_size, patterns, cardinality)
            else:
                pattern = eval(pattern)
                if fault_type == SAME_FEATURE_MAP_SAME_ROW:
                    assert pattern[-1] <= output_size[2] * output_size[3]
                    random_feature_map = np.random.randint(0, output_size[1])
                    random_index = np.random.randint(0, output_size[2] * output_size[3] - pattern[-1])
                    indexes = [
                        random_index + offset
                        for offset in pattern
                        ]
                    return [
                        np.unravel_index(index, shape=output_size)
                        for index in indexes
                        ]
                elif fault_type == SAME_FEATURE_MAP_SAME_COLUMN:
                    assert pattern[-1] <= output_size[3]
                    random_feature_map = np.random.randint(0, output_size[1])
                    random_index = np.random.randint(0, output_size[2] * output_size[3] - pattern[-1] * output_size[3])
                    indexes = [
                        random_index + offset * output_size[3]
                        for offset in pattern
                        ]
                    return [
                        np.unravel_index(index, shape=output_size)
                        for index in indexes
                        ]
                elif fault_type == SAME_FEATURE_MAP_BLOCK:
                    # TODO da rivedere questo assert
                    assert pattern[-1] * 16 <= output_size[2] * output_size[3]
                    random_feature_map = np.random.randint(0, output_size[1])
                    random_index = np.random.randint(0, output_size[2] * output_size[3] - pattern[-1] * 16)
                    indexes = [
                        random_index + offset * output_size[3]
                        for offset in pattern
                        ]
                    return [
                        np.unravel_index(index, shape=output_size)
                        for index in indexes
                        ]
                elif fault_type == SAME_FEATURE_MAP_RANDOM:
                    random_feature_map = np.random.randint(0, output_size[1])
                    indexes = np.random.choice(output_size[2] * output_size[3], replace=False, size=cardinality)
                    return [
                        np.unravel_index(index + random_feature_map * output_size[2] * output_size[3],
                                         shape=output_size)
                        for index in indexes
                        ]
                elif fault_type == MULTIPLE_FEATURE_MAPS_BULLET_WAKE:
                    # assert pattern[-1] < output_size[1]
                    if pattern[-1] >= output_size[1]:
                        new_card = 0
                        for elem in pattern:
                            if elem < output_size[1]:
                                new_card += 1
                        pattern = pattern[:new_card]
                        self.__debugcardinality = new_card
                    starting_feature_map_index = np.random.randint(0, output_size[1] - pattern[-1])
                    random_index = np.random.randint(0, output_size[2] * output_size[3])
                    indexes = [
                        random_index + (starting_feature_map_index + offset) * output_size[2] * output_size[3]
                        for offset in pattern
                        ]
                    return [
                        np.unravel_index(index, shape=output_size)
                        for index in indexes
                        ]
                elif fault_type == MULTIPLE_FEATURE_MAPS_BLOCK:
                    if max_linear_index < 16 * pattern[-1]:
                        new_card = 0
                        for elem in pattern:
                            if max_linear_index > elem * 16:
                                new_card += 1
                        pattern = pattern[:new_card]
                        self.__debugcardinality = new_card
                    random_index = np.random.randint(0, max_linear_index - 16 * pattern[-1])
                    indexes = [
                        random_index + 16 * offset
                        for offset in pattern
                    ]
                    return [
                        np.unravel_index(index, shape=output_size)
                        for index in indexes
                    ]
                elif (
                        fault_type == MULTIPLE_FEATURE_MAPS_SHATTER_GLASS or
                        fault_type == MULTIPLE_FEATURE_MAPS_QUASI_SHATTER_GLASS
                ):
                    if pattern[-1][0] >= output_size[1]:
                        new_card = 0
                        for elem in pattern:
                            if elem[0] < output_size[1]:
                                new_card += 1
                        pattern = pattern[:new_card]
                    assert pattern[-1][0] < output_size[1]
                    min_x = 0
                    max_x = 0
                    for feature in pattern:
                        if feature[1][0] < min_x:
                            min_x = feature[1][0]
                        if feature[1][-1] > max_x:
                            max_x = feature[1][-1]
                    random_feature_map = np.random.randint(0, output_size[1] - pattern[-1][0])
                    # TODO controlla perché max < min a volte
                    random_index = np.random.randint(min_x + 1, output_size[2] * output_size[3] - max_x)
                    indexes = []
                    for feature_pattern in pattern:
                        for offset in feature_pattern[1]:
                            indexes.append(
                                random_index + offset + (random_feature_map + feature_pattern[0]) * output_size[2] *
                                output_size[3])
                            # TODO filtrare gli index rimuovendo quelli < 0
                    return [
                        np.unravel_index(index, shape=output_size)
                        for index in indexes
                        ]
                elif fault_type == MULTIPLE_FEATURE_MAPS_UNCATEGORIZED:
                    indexes = np.random.choice(max_linear_index, size=cardinality, replace=False)
                    return [
                        np.unravel_index(index, shape=output_size)
                        for index in indexes
                        ]


if __name__ == "__main__":
    with open("./injectable_sites.json", "r") as injectables_sites_json:
        injectables_sites = []
        for injectable_site in json.load(injectables_sites_json):
            injectables_sites.append(InjectableSite(OperatorType[injectable_site["type"]],
                                                    injectable_site["name"], injectable_site["size"]))

        pippo = InjectionSitesGenerator(injectables_sites)
        pippo.generate_random_injection_sites(10000)