import os
from importlib.util import find_spec

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import qml
import qnn_module
import roboflow
import tensorflow as tf
from pkg_resources import Requirement, resource_filename
from roboflow import Roboflow as rf
from tensorflow.keras.models import load_model

rf = Roboflow(api_key="5bIb6TyRGOcHZqFJr1hn")
project = rf.workspace("best-uheyr").project("quan-bspqb")
version = project.version(3)
dataset = version.download("tensorflow")


function = 'detect_face'
if find_spec('cvlib') is None:
    raise ModuleNotFoundError('cvlib package is required to use this function')

def qml_model(inputs, weights):
    """
    Applies a quantum circuit to the given inputs using the specified weights.

    Args:
        inputs (list): List of input values.
        weights (list): List of weight values.

    Returns:
        list: List of expectation values for Pauli-Z measurements on each wire.
    """
    return quantum_circuit(inputs, weights)

def evaluate_mode(validation_generator, save_model_path):
    """
    Evaluates the model using the validation generator and saves the model.

    Args:
        validation_generator: The validation generator.
        save_model_path: The path to save the model.

    Returns:
        None
    """
    # Load the model
    model = load_model('MobileNetSSD_deploy.caffemodel')
    # Get the validation loss and accuracy
    loss, accuracy = model.evaluate_generator(validation_generator)
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))

def main(nb_images=5):
    """
    The main function of the program.

    Args:
        nb_images (int): The number of images.

    Returns:
        None
    """
    # CNN model evaluate
    model = load_model('MobileNetSSD_deploy.caffemodel')

class SequentialModule(tf.Module):
    """
    A sequential module class.

    This class represents a sequential module in TensorFlow.

    Args:
        name (str): The name of the module.

    Attributes:
        dense_1 (Dense): The first dense layer.
        dense_2 (Dense): The second dense layer.
    """
    def __init__(self, name=None):
        super().__init__(name=name)
        self.dense_1 = Dense(in_features=3, out_features=3)
        self.dense_2 = Dense(in_features=3, out_features=2)

    def __call__(self, x):
        x = self.dense_1(x)
        return self.dense_2(x)

class Dense(tf.Module):
    """
    A dense layer class.

    This class represents a dense layer in TensorFlow.

    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        name (str): The name of the layer.
    """
    def __init__(self, in_features, out_features, name=None):
        super().__init__(name=name)
        self.w = tf.Variable(tf.random.normal([in_features, out_features]), name='w')
        self.b = tf.Variable(tf.zeros([out_features]), name='b')

    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)

def getTestData():
    """
    Gets the test data.

    Returns:
        tuple: A tuple containing the test data.
    """
    x = tf.constant(testData.data)
    y = tf.constant(testData.target)
    return x, y

def quantum_circuit(inputs, weights):
    """
    Applies a quantum circuit to the given inputs using the specified weights.

    Args:
        inputs (list): List of input values.
        weights (list): List of weight values.

    Returns:
        list: List of expectation values for Pauli-Z measurements on each wire.
    """
    qml.BasisState(inputs, wires=range(4))
    qml.Rot(weights[0], weights[1], weights[2], wires=0)
    qml.Rot(weights[3], weights[4], weights[5], wires=1)
    qml.Rot(weights[6], weights[7], weights[8], wires=2)
    qml.Rot(weights[9], weights[10], weights[11], wires=3)
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

def qnn_model(inputs, weights):
    """
    Applies a quantum circuit to the given inputs using the specified weights.

    Args:
        inputs (list): List of input values.
        weights (list): List of weight values.

    Returns:
        list: List of expectation values for Pauli-Z measurements on each wire.
    """
    return quantum_circuit(inputs, weights)

def cost(weights, inputs, targets):
    """
    Calculates the cost function.

    Args:
        weights (list): List of weight values.
        inputs (list): List of input values.
        targets (list): List of target values.

    Returns:
        float: The cost value.
    """
    predictions = np.array([qnn_model(inp, weights) for inp in inputs])
    return np.mean((predictions - targets) ** 2)

def train_qnn(inputs, targets, steps=100, stepsize=0.1):
    """
    Trains the quantum model.

    Args:
        inputs (list): List of input values.
        targets (list): List of target values.
        steps (int): The number of training steps.
        stepsize (float): The step size for optimization.

    Returns:
        None
    """
    weights = np.random.random(size=(12,))
    opt = qml.AdamOptimizer(stepsize)
    
    for i in range(steps):
        weights, cost_val = opt.step_and_cost(lambda w: cost(w, inputs, targets), weights)
        if (i + 1) % 10 == 0:
            print("Step {}, Cost: {:.4f}".format(i+1, cost_val))

class SequentialModule(tf.Module):
    """
    A sequential module class.

    This class represents a sequential module in TensorFlow.

    Args:
        name (str): The name of the module.

    Attributes:
        dense_1 (Dense): The first dense layer.
        dense_2 (Dense): The second dense layer.
    """
    def __init__(self, name=None):
        super().__init__(name=name)
        self.dense_1 = Dense(in_features=3, out_features=3)
        self.dense_2 = Dense(in_features=3, out_features=2)

def detect_face(image, threshold=0.5, enable_gpu=False):
    """
    Detects faces in an image.

    Args:
        image (ndarray): The input image.
        threshold (float): The confidence threshold for face detection.
        enable_gpu (bool): Whether to enable GPU acceleration.

    Returns:
        tuple: A tuple containing the detected faces and their corresponding confidences.
    """
    if image is None:
        return None

    global is_initialized
    global prototxt
    global caffemodel
    global net
    
    if not is_initialized:

        # access resource files inside package
        prototxt = resource_filename(Requirement.parse('cvlib'),
        'cvlib' + '/' + 'data' + '/' + 'deploy.prototxt')
        caffemodel = resource_filename(Requirement.parse('cvlib'),
                                                'cvlib' + os.path.sep + 'data' + os.path.sep + 'res10_300x300_ssd_iter_140000.caffemodel')
        
        # read pre-trained wieights
        net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    
        is_initialized = True

    # enable GPU if requested
    if enable_gpu:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    (h, w) = image.shape[:2]

    # preprocessing input image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300,300), (104.0,177.0,123.0))
    net.setInput(blob)

    # apply face detection
    detections = net.forward()

    faces = []
    confidences = []

    # loop through detected faces
    for i in range(0, detections.shape[2]):
        conf = detections[0,0,i,2]

        # ignore detections with low confidence
        if conf < threshold:
            continue

        # get corner points of face rectangle
        box = detections[0,0,i,3:7] * np.array([w,h,w,h])
        (startX, startY, endX, endY) = box.astype('int')

        faces.append([startX, startY, endX, endY])
        confidences.append(conf)

    # return all detected faces and
    # corresponding confidences    
    return faces, confidences