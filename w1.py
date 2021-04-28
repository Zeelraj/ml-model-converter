from flask import Flask,render_template, make_response
import keras2onnx
from tensorflow import keras
import tensorflow as tf
import onnx
from google.protobuf.json_format import MessageToJson
import json
from torch.autograd import Variable
import torch.onnx as torch_onnx
from onnx_tf.backend import prepare
import torch
from keras.utils import plot_model
import torch.nn as nn
from torchviz import make_dot
import tensorflowjs as tfjs
dtype = torch.FloatTensor
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.h1 = nn.Linear(6, 10)
        self.ol = nn.Linear(6, 1)
        self.relu=nn.ReLU()
        self.softmax = nn.LogSoftmax()
    def forward(self, x):
     hidden = self.h1(x)
     output = self.ol(x)
     activation = self.relu(hidden)
     activation1 = self.softmax(output)
     return output
net = Net()
app=Flask(__name__)
@app.route('/')
def index():
    return render_template('Index.html')
@app.route("/kconversion/",methods=['POST'])
def kconversion():
    model=keras.models.load_model('model_keras')
    plot_model(model, to_file="model.png",show_shapes=True,show_layer_names=True,rankdir='TB',expand_nested=True,dpi=96)
    onnx_model = keras2onnx.convert_keras(model, 'model0.onnx', debug_mode=True)
    output_model_path = "./model0.onnx"
    # and save the model in ONNX format
    keras2onnx.save_model(onnx_model, output_model_path)
    onnx_model = onnx.load("model0.onnx")
    s = MessageToJson(onnx_model)
    onnx_json = json.loads(s)
    # Convert JSON to String
    onnx_str = json.dumps(onnx_json)
    with open("model1.json", "w") as json_file:
        json_file.write(onnx_str)
    resp = make_response(onnx_str)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp
@app.route("/tfconversion/",methods=['POST'])
def tfconversion():
    model = keras.models.load_model('model_keras')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
    tfjs.converters.save_keras_model(model,'/home/vatsal/Desktop/projects/open-api-workflow')
    file_name = "/home/vatsal/Desktop/projects/open-api-workflow/model.json"
    with open(file_name, 'r') as f:
        s1 = json.loads(f.read())
    s1_str = json.dumps(s1)
    resp = make_response(s1_str)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp
@app.route("/ptconversion/",methods=['POST'])
def ptconversion():
    model=torch.load('entire_model.pt')
    x = Variable(torch.randn(1,10,6))
    dot= make_dot(model(x), params=dict(model.named_parameters()))
    dot.format = 'png'
    dot.render('torchviz-sample')
    model.eval()
    input_shape = (1, 10, 6)
    model_onnx_path = "torch_model.onnx"
    dummy_input = Variable(torch.randn(1, *input_shape).type(dtype), requires_grad=True)
     # plot graph of variable, not of a nn.Module
    output = torch_onnx.export(net, dummy_input, model_onnx_path, verbose=True)
          # plot graph of variable, not of a nn.Module
    print("Export of torch_model.onnx complete!")
    onnx_model = onnx.load("torch_model.onnx")
    s = MessageToJson(onnx_model)
    onnx_json = json.loads(s)
    # Convert JSON to String
    onnx_str = json.dumps(onnx_json)
    with open("model2.json", "w") as json_file:
        json_file.write(onnx_str)
    resp = make_response(onnx_str)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp
@app.route("/tpconversion/",methods=['POST'])
def tpconversion():
    onnx_model = onnx.load("torch_model.onnx")
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph("/home/vatsal/Desktop/projects/open-api-workflow/model.pb")
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model("/home/vatsal/Desktop/projects/open-api-workflow/model.pb")  # path to the SavedModel directory
    tflite_model = converter.convert()
    # Save the model
    with open('model2.tflite', 'wb') as f:
        f.write(tflite_model)
    tfjs.converters.convert_tf_saved_model('/home/vatsal/Desktop/projects/open-api-workflow/model.pb','/home/vatsal/Desktop/projects/open-api-workflow/model3.json')
    file_name = "/home/vatsal/Desktop/projects/open-api-workflow/model3.json/model.json"
    with open(file_name, 'r') as f:
        s1 = json.loads(f.read())
    s1_str=json.dumps(s1)
    resp = make_response(s1_str)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp
if __name__=='__main__':
    app.run(debug=True)

