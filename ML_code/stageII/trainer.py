from __future__ import division
from __future__ import print_function

import prettytensor as pt
import tensorflow as tf
import numpy as np
import scipy.misc
import os
import sys
from six.moves import range
from progressbar import ETA, Bar, Percentage, ProgressBar
from PIL import Image, ImageDraw, ImageFont


from misc.config import cfg
from misc.utils import mkdir_p

TINY = 1e-8


def KL_loss(mu, log_sigma):
    with tf.name_scope("KL_divergence"):
        loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mu))
        loss = tf.reduce_mean(loss)
        return loss


class CondGANTrainer(object):
    def __init__(self,
                 model,
                 dataset=None,
                 exp_name="model",
                 ckt_logs_dir="ckt_logs",
                 ):
        self.model = model
        self.dataset = dataset
        self.exp_name = exp_name
        self.log_dir = ckt_logs_dir
        self.checkpoint_dir = ckt_logs_dir

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        self.model_path = cfg.TRAIN.PRETRAINED_MODEL

        self.log_vars = []

        self.hr_image_shape = self.dataset.image_shape
        ratio = self.dataset.hr_lr_ratio
        self.lr_image_shape = [int(self.hr_image_shape[0] / ratio),
                               int(self.hr_image_shape[1] / ratio),
                               self.hr_image_shape[2]]
        print('hr_image_shape', self.hr_image_shape)
        print('lr_image_shape', self.lr_image_shape)

    def build_placeholder(self):
        self.hr_images = tf.placeholder(
            tf.float32, [self.batch_size] + self.hr_image_shape,
            name='real_hr_images')
        self.hr_wrong_images = tf.placeholder(
            tf.float32, [self.batch_size] + self.hr_image_shape,
            name='wrong_hr_images'
        )
        self.embeddings = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.embedding_shape,
            name='conditional_embeddings'
        )

        self.generator_lr = tf.placeholder(
            tf.float32, [],
            name='generator_learning_rate'
        )
        self.discriminator_lr = tf.placeholder(
            tf.float32, [],
            name='discriminator_learning_rate'
        )
        #
        self.images = tf.image.resize_bilinear(self.hr_images,
                                               self.lr_image_shape[:2])
        self.wrong_images = tf.image.resize_bilinear(self.hr_wrong_images,
                                                     self.lr_image_shape[:2])

    def sample_encoded_context(self, embeddings):
        c_mean_logsigma = self.model.generate_condition(embeddings)
        mean = c_mean_logsigma[0]
        if cfg.TRAIN.COND_AUGMENTATION:
            epsilon = tf.truncated_normal(tf.shape(mean))
            stddev = tf.exp(c_mean_logsigma[1])
            c = mean + stddev * epsilon

            kl_loss = KL_loss(c_mean_logsigma[0], c_mean_logsigma[1])
        else:
            c = mean
            kl_loss = 0
        return c, cfg.TRAIN.COEFF.KL * kl_loss

    def init_opt(self):
        self.build_placeholder()

        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("g_net"):
                c, kl_loss = self.sample_encoded_context(self.embeddings)
                z = tf.random_normal([self.batch_size, cfg.Z_DIM])
                self.log_vars.append(("hist_c", c))
                self.log_vars.append(("hist_z", z))
                fake_images = self.model.get_generator(tf.concat(1, [c, z]))

            discriminator_loss, generator_loss =\
                self.compute_losses(self.images,
                                    self.wrong_images,
                                    fake_images,
                                    self.embeddings,
                                    flag='lr')
            generator_loss += kl_loss
            self.log_vars.append(("g_loss_kl_loss", kl_loss))
            self.log_vars.append(("g_loss", generator_loss))
            self.log_vars.append(("d_loss", discriminator_loss))

            with tf.variable_scope("hr_g_net"):
                hr_c, hr_kl_loss = self.sample_encoded_context(self.embeddings)
                self.log_vars.append(("hist_hr_c", hr_c))
                hr_fake_images = self.model.hr_get_generator(fake_images, hr_c)
            hr_discriminator_loss, hr_generator_loss =\
                self.compute_losses(self.hr_images,
                                    self.hr_wrong_images,
                                    hr_fake_images,
                                    self.embeddings,
                                    flag='hr')
            hr_generator_loss += hr_kl_loss
            self.log_vars.append(("hr_g_loss", hr_generator_loss))
            self.log_vars.append(("hr_d_loss", hr_discriminator_loss))

            self.prepare_trainer(discriminator_loss, generator_loss,
                                 hr_discriminator_loss, hr_generator_loss)
            self.define_summaries()

        with pt.defaults_scope(phase=pt.Phase.test):
            self.sampler()
            self.visualization(cfg.TRAIN.NUM_COPY)
            print("success")

    def sampler(self):
        with tf.variable_scope("g_net", reuse=True):
            c, _ = self.sample_encoded_context(self.embeddings)
            z = tf.random_normal([self.batch_size, cfg.Z_DIM])
            self.fake_images = self.model.get_generator(tf.concat(1, [c, z]))
        with tf.variable_scope("hr_g_net", reuse=True):
            hr_c, _ = self.sample_encoded_context(self.embeddings)
            self.hr_fake_images =\
                self.model.hr_get_generator(self.fake_images, hr_c)

    def compute_losses(self, images, wrong_images,
                       fake_images, embeddings, flag='lr'):
        if flag == 'lr':
            real_logit =\
                self.model.get_discriminator(images, embeddings)
            wrong_logit =\
                self.model.get_discriminator(wrong_images, embeddings)
            fake_logit =\
                self.model.get_discriminator(fake_images, embeddings)
        else:
            real_logit =\
                self.model.hr_get_discriminator(images, embeddings)
            wrong_logit =\
                self.model.hr_get_discriminator(wrong_images, embeddings)
            fake_logit =\
                self.model.hr_get_discriminator(fake_images, embeddings)

        real_d_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(real_logit,
                                                    tf.ones_like(real_logit))
        real_d_loss = tf.reduce_mean(real_d_loss)
        wrong_d_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(wrong_logit,
                                                    tf.zeros_like(wrong_logit))
        wrong_d_loss = tf.reduce_mean(wrong_d_loss)
        fake_d_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(fake_logit,
                                                    tf.zeros_like(fake_logit))
        fake_d_loss = tf.reduce_mean(fake_d_loss)
        if cfg.TRAIN.B_WRONG:
            discriminator_loss =\
                real_d_loss + (wrong_d_loss + fake_d_loss) / 2.
        else:
            discriminator_loss = real_d_loss + fake_d_loss
        if flag == 'lr':
            self.log_vars.append(("d_loss_real", real_d_loss))
            self.log_vars.append(("d_loss_fake", fake_d_loss))
            if cfg.TRAIN.B_WRONG:
                self.log_vars.append(("d_loss_wrong", wrong_d_loss))
        else:
            self.log_vars.append(("hr_d_loss_real", real_d_loss))
            self.log_vars.append(("hr_d_loss_fake", fake_d_loss))
            if cfg.TRAIN.B_WRONG:
                self.log_vars.append(("hr_d_loss_wrong", wrong_d_loss))

        generator_loss = \
            tf.nn.sigmoid_cross_entropy_with_logits(fake_logit,
                                                    tf.ones_like(fake_logit))
        generator_loss = tf.reduce_mean(generator_loss)
        if flag == 'lr':
            self.log_vars.append(("g_loss_fake", generator_loss))
        else:
            self.log_vars.append(("hr_g_loss_fake", generator_loss))

        return discriminator_loss, generator_loss

    def define_one_trainer(self, loss, learning_rate, key_word):
        '''Helper function for init_opt'''
        all_vars = tf.trainable_variables()
        tarin_vars = [var for var in all_vars if
                      var.name.startswith(key_word)]

        opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
        trainer = pt.apply_optimizer(opt, losses=[loss], var_list=tarin_vars)
        return trainer

    def prepare_trainer(self, discriminator_loss, generator_loss,
                        hr_discriminator_loss, hr_generator_loss):
        ft_lr_retio = cfg.TRAIN.FT_LR_RETIO
        self.discriminator_trainer =\
            self.define_one_trainer(discriminator_loss,
                                    self.discriminator_lr * ft_lr_retio,
                                    'd_')
        self.generator_trainer =\
            self.define_one_trainer(generator_loss,
                                    self.generator_lr * ft_lr_retio,
                                    'g_')
        self.hr_discriminator_trainer =\
            self.define_one_trainer(hr_discriminator_loss,
                                    self.discriminator_lr,
                                    'hr_d_')
        self.hr_generator_trainer =\
            self.define_one_trainer(hr_generator_loss,
                                    self.generator_lr,
                                    'hr_g_')

        self.ft_generator_trainer = \
            self.define_one_trainer(hr_generator_loss,
                                    self.generator_lr * cfg.TRAIN.FT_LR_RETIO,
                                    'g_')

        self.log_vars.append(("hr_d_learning_rate", self.discriminator_lr))
        self.log_vars.append(("hr_g_learning_rate", self.generator_lr))

    def define_summaries(self):
        all_sum = {'g': [], 'd': [], 'hr_g': [], 'hr_d': [], 'hist': []}
        for k, v in self.log_vars:
            if k.startswith('g'):
                all_sum['g'].append(tf.scalar_summary(k, v))
            elif k.startswith('d'):
                all_sum['d'].append(tf.scalar_summary(k, v))
            elif k.startswith('hr_g'):
                all_sum['hr_g'].append(tf.scalar_summary(k, v))
            elif k.startswith('hr_d'):
                all_sum['hr_d'].append(tf.scalar_summary(k, v))
            elif k.startswith('hist'):
                all_sum['hist'].append(tf.histogram_summary(k, v))

        self.g_sum = tf.merge_summary(all_sum['g'])
        self.d_sum = tf.merge_summary(all_sum['d'])
        self.hr_g_sum = tf.merge_summary(all_sum['hr_g'])
        self.hr_d_sum = tf.merge_summary(all_sum['hr_d'])
        self.hist_sum = tf.merge_summary(all_sum['hist'])

    def visualize_one_superimage(self, img_var, images, rows, filename):
        stacked_img = []
        for row in range(rows):
            img = images[row * rows, :, :, :]
            row_img = [img]  
            for col in range(rows):
                row_img.append(img_var[row * rows + col, :, :, :])
            stacked_img.append(tf.concat(1, row_img))
        imgs = tf.expand_dims(tf.concat(0, stacked_img), 0)
        current_img_summary = tf.image_summary(filename, imgs)
        return current_img_summary, imgs

    def visualization(self, n):
        fake_sum_train, superimage_train =\
            self.visualize_one_superimage(self.fake_images[:n * n],
                                          self.images[:n * n],
                                          n, "train")
        fake_sum_test, superimage_test =\
            self.visualize_one_superimage(self.fake_images[n * n:2 * n * n],
                                          self.images[n * n:2 * n * n],
                                          n, "test")
        self.superimages = tf.concat(0, [superimage_train, superimage_test])
        self.image_summary = tf.merge_summary([fake_sum_train, fake_sum_test])

        hr_fake_sum_train, hr_superimage_train =\
            self.visualize_one_superimage(self.hr_fake_images[:n * n],
                                          self.hr_images[:n * n, :, :, :],
                                          n, "hr_train")
        hr_fake_sum_test, hr_superimage_test =\
            self.visualize_one_superimage(self.hr_fake_images[n * n:2 * n * n],
                                          self.hr_images[n * n:2 * n * n],
                                          n, "hr_test")
        self.hr_superimages =\
            tf.concat(0, [hr_superimage_train, hr_superimage_test])
        self.hr_image_summary =\
            tf.merge_summary([hr_fake_sum_train, hr_fake_sum_test])

    def preprocess(self, x, n):
        for i in range(n):
            for j in range(1, n):
                x[i * n + j] = x[i * n]
        return x

    def epoch_sum_images(self, sess, n):
        images_train, _, embeddings_train, captions_train, _ =\
            self.dataset.train.next_batch(n * n, cfg.TRAIN.NUM_EMBEDDING)
        images_train = self.preprocess(images_train, n)
        embeddings_train = self.preprocess(embeddings_train, n)

        images_test, _, embeddings_test, captions_test, _ =\
            self.dataset.test.next_batch(n * n, 1)
        images_test = self.preprocess(images_test, n)
        embeddings_test = self.preprocess(embeddings_test, n)

        images = np.concatenate([images_train, images_test], axis=0)
        embeddings =\
            np.concatenate([embeddings_train, embeddings_test], axis=0)

        if self.batch_size > 2 * n * n:
            images_pad, _, embeddings_pad, _, _ =\
                self.dataset.test.next_batch(self.batch_size - 2 * n * n, 1)
            images = np.concatenate([images, images_pad], axis=0)
            embeddings = np.concatenate([embeddings, embeddings_pad], axis=0)

        feed_out = [self.superimages, self.image_summary,
                    self.hr_superimages, self.hr_image_summary]
        feed_dict = {self.hr_images: images,
                     self.embeddings: embeddings}
        gen_samples, img_summary, hr_gen_samples, hr_img_summary =\
            sess.run(feed_out, feed_dict)

        scipy.misc.imsave('%s/lr_fake_train.jpg' %
                          (self.log_dir), gen_samples[0])
        scipy.misc.imsave('%s/lr_fake_test.jpg' %
                          (self.log_dir), gen_samples[1])
        #
        scipy.misc.imsave('%s/hr_fake_train.jpg' %
                          (self.log_dir), hr_gen_samples[0])
        scipy.misc.imsave('%s/hr_fake_test.jpg' %
                          (self.log_dir), hr_gen_samples[1])

        pfi_test = open(self.log_dir + "/test.txt", "w")
        for row in range(n):

            pfi_test.write('\n***row %d***\n' % row)
            pfi_test.write(captions_test[row * n])
        pfi_test.close()

        return img_summary, hr_img_summary

    def build_model(self, sess):
        self.init_opt()

        sess.run(tf.initialize_all_variables())
        if len(self.model_path) > 0:
            print("Reading model parameters from %s" % self.model_path)
            all_vars = tf.trainable_variables()
            restore_vars = []
            for var in all_vars:
                if var.name.startswith('g_') or var.name.startswith('d_'):
                    restore_vars.append(var)
            saver = tf.train.Saver(restore_vars)
            saver.restore(sess, self.model_path)

            istart = self.model_path.rfind('_') + 1
            iend = self.model_path.rfind('.')
            counter = self.model_path[istart:iend]
            counter = int(counter)
        else:
            print("Created model with fresh parameters.")
            counter = 0
        return counter

    def train_one_step(self, generator_lr,
                       discriminator_lr,
                       counter, summary_writer, log_vars, sess):
        hr_images, hr_wrong_images, embeddings, _, _ =\
            self.dataset.train.next_batch(self.batch_size,
                                          cfg.TRAIN.NUM_EMBEDDING)
        feed_dict = {self.hr_images: hr_images,
                     self.hr_wrong_images: hr_wrong_images,
                     self.embeddings: embeddings,
                     self.generator_lr: generator_lr,
                     self.discriminator_lr: discriminator_lr
                     }
        if cfg.TRAIN.FINETUNE_LR:
            feed_out_d = [self.hr_discriminator_trainer,
                          self.hr_d_sum,
                          log_vars,
                          self.hist_sum]
            ret_list = sess.run(feed_out_d, feed_dict)
            summary_writer.add_summary(ret_list[1], counter)
            log_vals = ret_list[2]
            summary_writer.add_summary(ret_list[3], counter)
            feed_out_g = [self.hr_generator_trainer,
                          self.ft_generator_trainer,
                          self.hr_g_sum]
            _, _, hr_g_sum = sess.run(feed_out_g, feed_dict)
            summary_writer.add_summary(hr_g_sum, counter)
            feed_out_d = [self.discriminator_trainer, self.d_sum]
            _, d_sum = sess.run(feed_out_d, feed_dict)
            summary_writer.add_summary(d_sum, counter)
            feed_out_g = [self.generator_trainer, self.g_sum]
            _, g_sum = sess.run(feed_out_g, feed_dict)
            summary_writer.add_summary(g_sum, counter)
        else:
            feed_out_d = [self.hr_discriminator_trainer,
                          self.hr_d_sum,
                          log_vars,
                          self.hist_sum]
            ret_list = sess.run(feed_out_d, feed_dict)
            summary_writer.add_summary(ret_list[1], counter)
            log_vals = ret_list[2]
            summary_writer.add_summary(ret_list[3], counter)
            feed_out_g = [self.hr_generator_trainer,
                          self.hr_g_sum]
            _, hr_g_sum = sess.run(feed_out_g, feed_dict)
            summary_writer.add_summary(hr_g_sum, counter)

        return log_vals

    def train(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:%d" % cfg.GPU_ID):
                counter = self.build_model(sess)
                saver = tf.train.Saver(tf.all_variables(),
                                       keep_checkpoint_every_n_hours=5)

                summary_writer = tf.train.SummaryWriter(self.log_dir,
                                                        sess.graph)

                if cfg.TRAIN.FINETUNE_LR:
                    keys = ["hr_d_loss", "hr_g_loss", "d_loss", "g_loss"]
                else:
                    keys = ["d_loss", "g_loss"]
                log_vars = []
                log_keys = []
                for k, v in self.log_vars:
                    if k in keys:
                        log_vars.append(v)
                        log_keys.append(k)
                generator_lr = cfg.TRAIN.GENERATOR_LR
                discriminator_lr = cfg.TRAIN.DISCRIMINATOR_LR
                lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH
                number_example = self.dataset.train._num_examples
                updates_per_epoch = int(number_example / self.batch_size)
                decay_start = cfg.TRAIN.PRETRAINED_EPOCH
                epoch_start = int(counter / updates_per_epoch)
                for epoch in range(epoch_start, self.max_epoch):
                    widgets = ["epoch #%d|" % epoch,
                               Percentage(), Bar(), ETA()]
                    pbar = ProgressBar(maxval=updates_per_epoch,
                                       widgets=widgets)
                    pbar.start()

                    if epoch % lr_decay_step == 0 and epoch > decay_start:
                        generator_lr *= 0.5
                        discriminator_lr *= 0.5

                    all_log_vals = []
                    for i in range(updates_per_epoch):
                        pbar.update(i)
                        log_vals = self.train_one_step(generator_lr,
                                                       discriminator_lr,
                                                       counter, summary_writer,
                                                       log_vars, sess)
                        all_log_vals.append(log_vals)
                        counter += 1
                        if counter % self.snapshot_interval == 0:
                            snapshot_path = "%s/%s_%s.ckpt" %\
                                             (self.checkpoint_dir,
                                              self.exp_name,
                                              str(counter))
                            fn = saver.save(sess, snapshot_path)
                            print("Model saved in file: %s" % fn)

                    img_summary, img_summary2 =\
                        self.epoch_sum_images(sess, cfg.TRAIN.NUM_COPY)
                    summary_writer.add_summary(img_summary, counter)
                    summary_writer.add_summary(img_summary2, counter)

                    avg_log_vals = np.mean(np.array(all_log_vals), axis=0)
                    dic_logs = {}
                    for k, v in zip(log_keys, avg_log_vals):
                        dic_logs[k] = v

                    log_line = "; ".join("%s: %s" %
                                         (str(k), str(dic_logs[k]))
                                         for k in dic_logs)
                    print("Epoch %d | " % (epoch) + log_line)
                    sys.stdout.flush()
                    if np.any(np.isnan(avg_log_vals)):
                        raise ValueError("NaN detected!")
