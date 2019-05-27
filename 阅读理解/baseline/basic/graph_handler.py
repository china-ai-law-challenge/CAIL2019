import gzip
import json
import os
import pickle
import tensorflow as tf
from basic.evaluator import Evaluation
from my.utils import short_floats


class GraphHandler(object):
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.saver = tf.train.Saver(max_to_keep=config.max_to_keep)
        self.writer = None

    def initialize(self, sess):
        sess.run(tf.global_variables_initializer())
        if self.config.load:
            self._load(sess)

        if self.config.mode == 'Tr':
            self.writer = tf.summary.FileWriter(self.config.log_dir, graph=tf.get_default_graph())

    def save(self, sess, save_path=None):
        saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        if save_path is None:
            save_path = os.path.join(self.config.save_dir, "best.ckpt")
        saver.save(sess, save_path)

    def _load(self, sess):
        config = self.config
        vars_ = {var.name.split(":")[0]: var for var in tf.global_variables()}
        if config.load_ema:
            ema = self.model.var_ema
            for var in tf.trainable_variables():
                del vars_[var.name.split(":")[0]]
                vars_[ema.average_name(var)] = var
        saver = tf.train.Saver(vars_, max_to_keep=config.max_to_keep)

        if config.load_path:
            save_path = config.load_path
        elif config.load_step > 0:
            save_path = os.path.join(config.save_dir, "{}-{}".format(config.model_name, config.load_step))
        else:
            save_dir = config.save_dir
            checkpoint = tf.train.get_checkpoint_state(save_dir)
            assert checkpoint is not None, "cannot load checkpoint at {}".format(save_dir)
            save_path = checkpoint.model_checkpoint_path
        print("Loading saved model from {}".format(save_path))
        saver.restore(sess, save_path)

    def add_summary(self, summary, global_step):
        self.writer.add_summary(summary, global_step)

    def add_summaries(self, summaries, global_step):
        for summary in summaries:
            self.add_summary(summary, global_step)

    def dump_eval(self, e, precision=2, path=None):
        assert isinstance(e, Evaluation)
        if self.config.dump_pickle:
            path = path or os.path.join(self.config.eval_dir, "{}-{}.pklz".format(e.data_type, str(e.global_step).zfill(6)))
            with gzip.open(path, 'wb', compresslevel=3) as fh:
                pickle.dump(e.dict, fh)
        else:
            path = path or os.path.join(self.config.eval_dir, "{}-{}.json".format(e.data_type, str(e.global_step).zfill(6)))
            with open(path, 'w') as fh:
                json.dump(short_floats(e.dict, precision), fh, ensure_ascii=False)
    
    def dump_answer(self, e, path=None):
        assert isinstance(e, Evaluation)
        if self.config.mode == "Tr":
            path = path or os.path.join(self.config.answer_dir, "{}-{}.json".format(e.data_type, str(e.global_step).zfill(6)))
        else:
            path = path or os.path.join(self.config.out_dir, "result.json")
        with open(path, 'w') as fh:
            result = [{"id": key, "answer": val} for key, val in e.id2answer_dict.items() if key != "scores" and key != "na"]
            json.dump(result, fh)

