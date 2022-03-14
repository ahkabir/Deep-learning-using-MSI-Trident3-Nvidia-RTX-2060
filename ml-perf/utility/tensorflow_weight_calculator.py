import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util


#def create_graph(modelFullPath):
#    """Creates a graph from saved GraphDef file and returns a saver."""
#    # Creates graph from saved graph_def.pb.
#    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
#        graph_def = tf.GraphDef()
#        graph_def.ParseFromString(f.read())
#        tf.import_graph_def(graph_def, name='')

#GRAPH_DIR='/home/ahsan/model_zoo/tensorflow_models/caffe/alexnet/alexnet.pb'
#create_graph(GRAPH_DIR)

#constant_values = {}

#with tf.Session() as sess:
#  constant_ops = [op for op in sess.graph.get_operations() if op.type == "Const"]
#  for constant_op in constant_ops:
#    constant_values[constant_op.name] = sess.run(constant_op.outputs[0])
#    print (constant_op.name)




# solution 2

#GRAPH_PB_PATH='./model/tensorflow_inception_v3_stripped_optimized_quantized.pb' #path to your .pb file
#GRAPH_PB_PATH='/home/ahsan/model_zoo/tensorflow_models/caffe/alexnet/alexnet.pb'
GRAPH_PB_PATH='./alexnet_frozen.pb'
with tf.Session() as sess:
  print("load graph")
  with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    graph_nodes=[n for n in graph_def.node]
    wts = [n for n in graph_nodes if n.op=='Const']
    for n in wts:
        print "Name of the node - %s" % n.name
        print "Value - " 
        print tensor_util.MakeNdarray(n.attr['value'].tensor)
