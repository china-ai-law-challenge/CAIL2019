import argparse
from basic.freeze_model import freeze_graph
import math
import os
import shutil
from pprint import pprint
import time
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import sys
import json
from basic.evaluator import ForwardEvaluator, MultiGPUF1Evaluator
from basic.graph_handler import GraphHandler
from basic.model import get_multi_gpu_models
from basic.trainer import MultiGPUTrainer
from basic.read_data import read_data, get_squad_data_filter, update_config
from my.tensorflow import get_num_params
from memory_profiler import profile

def set_dirs(config):
    # create directories
    assert config.load or config.mode == 'Tr', "config.load must be True if not training"
    if not config.load and os.path.exists(config.out_dir):
        shutil.rmtree(config.out_dir)
    
    if config.mode == "Tr":
        config.save_dir = os.path.join(config.out_dir, "save")
        config.log_dir = os.path.join(config.out_dir, "log")
        config.eval_dir = os.path.join(config.out_dir, "eval")
        config.answer_dir = os.path.join(config.out_dir, "answer")
        config.timeline_dir = os.path.join(config.out_dir, "timeline")
        if not os.path.exists(config.out_dir):
            os.makedirs(config.out_dir)
        if not os.path.exists(config.save_dir):
            os.mkdir(config.save_dir)
        if not os.path.exists(config.log_dir):
            os.mkdir(config.log_dir)
        if not os.path.exists(config.answer_dir):
            os.mkdir(config.answer_dir)
        if not os.path.exists(config.eval_dir):
            os.mkdir(config.eval_dir)
        if not os.path.exists(config.timeline_dir):
            os.mkdir(config.timeline_dir)

def _config_debug(config):
    if config.debug:
        config.num_steps = 2
        config.eval_period = 1
        config.log_period = 1
        config.save_period = 1
        config.val_num_batches = 2
        config.test_num_batches = 2


def _train(config):
    data_filter = get_squad_data_filter(config)
    train_data = read_data(config, config.trainfile, "train", data_filter = data_filter)
    dev_data = read_data(config, config.validfile, "valid", data_filter = data_filter)
    update_config(config, [train_data, dev_data])

    _config_debug(config)

    models = get_multi_gpu_models(config)
    model = models[0]
    print("num params: {}".format(get_num_params()))
    trainer = MultiGPUTrainer(config, models)
    evaluator = MultiGPUF1Evaluator(config, models, tensor_dict=model.tensor_dict if config.vis else None)
    # controls all tensors and variables in the graph, including loading /saving
    graph_handler = GraphHandler(config, model)

    # Variables
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    graph_handler.initialize(sess)

    # Begin training
    num_steps = min(config.num_steps,int(math.ceil(train_data.num_examples /
                                                  (config.batch_size * config.num_gpus))) * config.num_epochs)
    acc = 0
    for batches in tqdm(train_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps,
                                                     shuffle=True, cluster=config.cluster),
                        total=num_steps):
        global_step = sess.run(model.global_step) + 1  # +1 because all calculations are done after step
        get_summary = global_step % config.log_period == 0
        loss, summary, train_op = trainer.step(sess, batches, get_summary=get_summary)
        if get_summary:
            graph_handler.add_summary(summary, global_step)

        # Occasional evaluation and saving
        if global_step % config.save_period == 0:
            num_steps = int(math.ceil(dev_data.num_examples / (config.batch_size * config.num_gpus)))
            if 0 < config.val_num_batches < num_steps:
                num_steps = config.val_num_batches
            e_train = evaluator.get_evaluation_from_batches(sess, tqdm(train_data.get_multi_batches(
                config.batch_size, config.num_gpus, num_steps=num_steps), total=num_steps))
            graph_handler.add_summaries(e_train.summaries, global_step)
            e_dev = evaluator.get_evaluation_from_batches(
                sess, tqdm(dev_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps),
                           total=num_steps))
            graph_handler.add_summaries(e_dev.summaries, global_step)

            if e_dev.acc > acc:
                acc = e_dev.acc
                print("begin saving model...")
                print(e_dev)
                graph_handler.save(sess)
                print("end saving model, dumping eval and answer...")
                if config.dump_eval:
                    graph_handler.dump_eval(e_dev)
                if config.dump_answer:
                    graph_handler.dump_answer(e_dev)
                print("end dumping")

    print("begin freezing model...")

    config.clear_device = False
    config.input_path = graph_handler.save_path
    config.output_path = "model"
    config.input_names = None
    config.output_names = None

    freeze_graph(config)
    print("model frozen at {}".format(config.output_path))


@profile
def _test(config):
    t1 = time.time()
    print("[{}] loading data..".format(t1))
    test_data = read_data(config, config.testfile, "test")
    t2 = time.time()
    print("[{}] updating config..".format(t2))
    update_config(config, [test_data])

    _config_debug(config)

    models = get_multi_gpu_models(config)
    model = models[0]
    evaluator = MultiGPUF1Evaluator(config, models, tensor_dict=models[0].tensor_dict if config.vis else None)
    graph_handler = GraphHandler(config, model)
    t3 = time.time()
    print("[{}] creating session..".format(t3))
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    t4 = time.time()
    print("[{}] initializing session..".format(t4))
    graph_handler.initialize(sess)
    num_steps = int(math.ceil(test_data.num_examples / (config.batch_size * config.num_gpus)))
    if 0 < config.test_num_batches < num_steps:
        num_steps = config.test_num_batches

    e = None
    t5 = time.time()
    print("loading model takes {}s\n begin evaluating..".format(t5 - t3))
    count = 0
    total_time = 0
    for multi_batch in tqdm(test_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps,
                                                        cluster=config.cluster), total=num_steps):
        t_start = time.time()
        evaluator.set_count(count)
        ei = evaluator.get_evaluation(sess, multi_batch)
        t_end = time.time()
        count += 1
        single_time = t_end - t_start
        total_time += single_time
        answer_id = list(ei.id2answer_dict["scores"].keys())[0]
        answer = ei.id2answer_dict[answer_id]
        print("id: {}, answer: {}, correct: {}, time: {:6.4f}s"
              .format(answer_id, answer.encode('ascii', 'ignore').decode('ascii'), int(ei.acc) == 1, single_time))
        sys.stdout.flush()
        e = ei if e is None else e + ei

    t6 = time.time()
    #print("[{}] finish evaluation".format(t6))
    #print("total time:{} for {} evaluations, avg:{}".format(total_time, count, total_time * 1.0 / count))
    
    print(e)
    print("dumping answer ...")
    graph_handler.dump_answer(e)
    """
    print("dumping eval ...")
    graph_handler.dump_eval(e)
    """

class Config:
    def __init__(self):
        self.config = "config"

def print_configuration_op(FLAGS):
    print('My Configurations:')
    config = Config()
    #pdb.set_trace()
    for name, value in FLAGS.__flags.items():
        value=value.value
        if type(value) == float:
            print(' %s:\t %f'%(name, value))
        elif type(value) == int:
            print(' %s:\t %d'%(name, value))
        elif type(value) == str:
            print(' %s:\t %s'%(name, value))
        elif type(value) == bool:
            print(' %s:\t %s'%(name, value))
        else:
            print('%s:\t %s' % (name, value))

        if not hasattr(config, name):
            setattr(config, name, value)

    return config

def main(_):
    config = flags.FLAGS
    config = print_configuration_op(config)
    set_dirs(config)
    with tf.device(config.device):
        if config.mode == 'Tr':
            if not config.trainfile or not config.validfile:
                raise ValueError("if mode is train, trainfile and validfile is needed.")
            _train(config)
        elif config.mode == 'Te':
            print(config.testfile)
            if not config.testfile:
                raise ValueError("if mode is test, testfile is needed.")
            _test(config)
        else:
            raise ValueError("invalid value for 'mode': {}".format(config.mode))

flags = tf.app.flags

# Names and directories
flags.DEFINE_string("model_name", "basic", "Model name [basic]")
flags.DEFINE_string("trainfile", None, "Train File [data]")
flags.DEFINE_string("validfile", None, "Valid File [data]")
flags.DEFINE_string("testfile", "../data/data.json", "Test File [data]")
flags.DEFINE_string("out_dir", "../result", "out base dir [data]")
flags.DEFINE_integer("test_size", 0, "Data size (number of questions) [0 means all]")
flags.DEFINE_string("run_id", "1", "Run ID [0]")
flags.DEFINE_string("answer_path", "", "Answer path []")
flags.DEFINE_string("eval_path", "", "Eval path []")
flags.DEFINE_string("load_path", "model/best.ckpt", "Load path []")
flags.DEFINE_string("shared_path", "shared.json", "Shared path []")
flags.DEFINE_bool("prof", True, "profiling testing? [True]")

# Device placement
flags.DEFINE_string("device", "/gpu:0", "default device for summing gradients. [/gpu:0]")
flags.DEFINE_string("device_type", "gpu", "device for computing gradients (parallelization). cpu | gpu [gpu]")
flags.DEFINE_integer("num_gpus", 1, "num of gpus or cpus for computing gradients [1]")

# Essential training and test options
flags.DEFINE_string("mode", "Te", "Train | Test")
flags.DEFINE_boolean("load", True, "load saved data? [True]")
flags.DEFINE_bool("single", False, "supervise only the answer sentence? [False]")
flags.DEFINE_boolean("debug", False, "Debugging mode? [False]")
flags.DEFINE_bool('load_ema', True, "load exponential average of variables when testing?  [True]")
# flags.DEFINE_bool("eval", True, "eval? [True]")
flags.DEFINE_bool("wy", False, "Use wy for loss / eval? [False]")
flags.DEFINE_bool("na", True, "Enable no answer strategy and learn bias? [False]")
flags.DEFINE_float("th", 0.5, "Threshold [0.5]")

# Training / test parameters
flags.DEFINE_integer("batch_size", 24, "Batch size [40]")
flags.DEFINE_integer("val_num_batches", 100, "validation num batches [100]")
flags.DEFINE_integer("test_num_batches", 0, "test num batches [0]")
flags.DEFINE_integer("num_epochs", 12, "Total number of epochs for training [12]")
flags.DEFINE_integer("num_steps", 25000, "Number of steps [25000]")
flags.DEFINE_integer("load_step", 0, "load step [0]")
flags.DEFINE_float("init_lr", 0.001, "Initial learning rate [0.001]")
flags.DEFINE_float("input_keep_prob", 0.8, "Input keep prob for the dropout of LSTM weights [0.8]")
flags.DEFINE_float("keep_prob", 0.8, "Keep prob for the dropout of Char-CNN weights [0.8]")
flags.DEFINE_float("wd", 0.0, "L2 weight decay for regularization [0.0]")
flags.DEFINE_integer("hidden_size", 100, "Hidden size [100]")
flags.DEFINE_integer("word_emb_size", 300, "word-level word embedding size [300]")
flags.DEFINE_integer("char_out_size", 100, "char-level word embedding size [100]")
flags.DEFINE_integer("char_emb_size", 8, "Char emb size [8]")
flags.DEFINE_string("out_channel_dims", "100", "Out channel dims of Char-CNN, separated by commas [100]")
flags.DEFINE_string("filter_heights", "5", "Filter heights of Char-CNN, separated by commas [5]")
flags.DEFINE_bool("finetune", True, "Finetune word embeddings? [False]")
flags.DEFINE_bool("highway", True, "Use highway? [True]")
flags.DEFINE_integer("highway_num_layers", 2, "highway num layers [2]")
flags.DEFINE_bool("share_cnn_weights", True, "Share Char-CNN weights [True]")
flags.DEFINE_bool("share_lstm_weights", True, "Share pre-processing (phrase-level) LSTM weights [True]")
flags.DEFINE_float("var_decay", 0.999, "Exponential moving average decay for variables [0.999]")


# Optimizations
flags.DEFINE_bool("cluster", False, "Cluster data for faster training [False]")
flags.DEFINE_bool("len_opt", True, "Length optimization? [False]")
flags.DEFINE_bool("cpu_opt", False, "CPU optimization? GPU computation can be slower [False]")

# Logging and saving options
flags.DEFINE_boolean("progress", True, "Show progress? [True]")
flags.DEFINE_integer("log_period", 100, "Log period [100]")
# flags.DEFINE_integer("eval_period", 100, "Eval period [1000]")
flags.DEFINE_integer("save_period", 1, "Save Period [1000]")
flags.DEFINE_integer("max_to_keep", 1, "Max recent saves to keep [1]")
flags.DEFINE_bool("dump_eval", True, "dump eval? [True]")
flags.DEFINE_bool("dump_answer", True, "dump answer? [True]")
flags.DEFINE_bool("vis", False, "output visualization numbers? [False]")
flags.DEFINE_bool("dump_pickle", True, "Dump pickle instead of json? [True]")
flags.DEFINE_float("decay", 0.9, "Exponential moving average decay for logging values [0.9]")

# Thresholds for speed and less memory usage
flags.DEFINE_integer("word_count_th", 10, "word count th [100]")
flags.DEFINE_integer("char_count_th", 50, "char count th [500]")
flags.DEFINE_integer("sent_size_th", 400, "sent size th [64]")
flags.DEFINE_integer("num_sents_th", 8, "num sents th [8]")
flags.DEFINE_integer("ques_size_th", 30, "ques size th [32]")
flags.DEFINE_integer("word_size_th", 16, "word size th [16]")
flags.DEFINE_integer("para_size_th", 256, "para size th [256]")

# Advanced training options
flags.DEFINE_bool("lower_word", True, "lower word [True]")
flags.DEFINE_bool("squash", False, "squash the sentences into one? [False]")
flags.DEFINE_bool("swap_memory", True, "swap memory? [True]")
flags.DEFINE_string("data_filter", "max", "max | valid | semi [max]")
flags.DEFINE_bool("use_glove_for_unk", False, "use glove for unk [False]")
flags.DEFINE_bool("known_if_glove", False, "consider as known if present in glove [False]")
flags.DEFINE_string("logit_func", "tri_linear", "logit func [tri_linear]")
flags.DEFINE_string("answer_func", "linear", "answer logit func [linear]")
flags.DEFINE_string("sh_logit_func", "tri_linear", "sh logit func [tri_linear]")

# Ablation options
flags.DEFINE_bool("use_char_emb", False, "use char emb? [True]")
flags.DEFINE_bool("use_word_emb", True, "use word embedding? [True]")
flags.DEFINE_bool("q2c_att", True, "question-to-context attention? [True]")
flags.DEFINE_bool("c2q_att", True, "context-to-question attention? [True]")
flags.DEFINE_bool("dynamic_att", False, "Dynamic attention [False]")


if __name__ == "__main__":
    tf.app.run()
