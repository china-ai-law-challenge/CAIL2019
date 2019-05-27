import argparse
import os
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference as opt_inference
from tensorflow.python.training import saver as saver_lib


def freeze_graph(config):
    input_names = config.input_names
    output_names = config.output_names
    if output_names is None:
        output_names = ["eval_concat/yp", "eval_concat/yp2", "eval_concat/wy", "eval_concat/loss"]

    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)

    frozen_model_path = os.path.join(config.output_path, "frozen_model.pb")
    checkpoint_file = config.input_path
    if not saver_lib.checkpoint_exists(checkpoint_file):
        print("Checkpoint file '" + checkpoint_file + "' doesn't exist!")
        exit(-1)

    print("begin loading model")
    saver = tf.train.import_meta_graph(checkpoint_file + '.meta', clear_devices=config.clear_device)

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver.restore(sess, checkpoint_file)
        print("model loaded")
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, input_graph_def, output_node_names=output_names)

        if input_names is not None:
            output_graph_def = opt_inference(output_graph_def, input_names,
                                             output_names, dtypes.float32.as_datatype_enum)

        with tf.gfile.GFile(frozen_model_path, "wb") as f:
            f.write(output_graph_def.SerializeToString())
            print("frozen graph binary saved to: {}".format(frozen_model_path))

        frozen_model_text_path = "{}.txt".format(frozen_model_path)
        with tf.gfile.FastGFile(frozen_model_text_path, "wb") as f:
            f.write(str(output_graph_def))
            print("frozen graph text saved to: {}".format(frozen_model_text_path))

        print("%d ops in the final graph." % len(output_graph_def.node))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear_device", action="store_false",
                        help="whether clear device or not, default is false")
    parser.add_argument("--input_path", type=str, default="data/basic/00/save/best.ckpt",
                        help="path to the output frozen model, default is `pwd`/data/basic/00/save/best.ckpt")
    parser.add_argument("--output_path", type=str, default="model",
                        help="path to the output frozen model, default is `pwd`/model/")
    parser.add_argument("--input_names", type=str, default=None,
                        help="input_names of the model, default is None")
    parser.add_argument("--output_names", type=str, default=None,
                        help="output_names of the model, default is None")
    args = parser.parse_args()
    freeze_graph(args)
