# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
from math import floor, ceil

from transformers import (WEIGHTS_NAME, BertConfig,
                          BertTokenizer,
                          BertModel,
                          RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer,
                          XLMConfig, XLMForSequenceClassification,
                          XLMTokenizer, XLNetConfig,
                          XLNetForSequenceClassification,
                          XLNetTokenizer,
                          DistilBertConfig,
                          DistilBertForSequenceClassification,
                          DistilBertTokenizer)

from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

# from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers.data.processors import glue_processors as processors
# from transformers import glue_convert_examples_to_features as convert_examples_to_features

from code.TextProcessing import clean_data
from code.models import BertForSequenceClassification
from code.metrics import compute_metrics
from code.AdversarialTraining import FGM, PGD

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig,
                                                                                RobertaConfig, DistilBertConfig)), ())


MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}


def write_file(datas, output_file):
    with open(output_file, 'w+', encoding='utf-8') as f:
        for obj in datas:
            json.dump(obj, f, ensure_ascii=False)
            f.write("\n")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a=None, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label


def read_examples(input_file, is_training):
    examples = []
    if is_training:
        df = pd.read_csv(input_file, delimiter="\t")
        df = clean_data(df, list(df.columns[[1, 2]]))
        for val in df.values:
            examples.append(InputExample(guid=0, text_a=val[1], text_b=val[2], label=val[0]))
    else:
        df = pd.read_csv(input_file, delimiter="\t")
        df = clean_data(df, list(df.columns[[1, 2]]))
        for val in df.values:
            examples.append(InputExample(guid=0, text_a=val[1], text_b=val[2], label=0))
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length, is_training):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for example_index, example in enumerate(examples):

        context_tokens = tokenizer.tokenize(example.text_a)
        ending_tokens_a = tokenizer.tokenize(example.text_b)

        choices_features = []
        context_tokens_choice = context_tokens
        q, a = _truncate_seq_pair(context_tokens_choice, ending_tokens_a, max_seq_length)
        tokens = ["[CLS]"] + q + ["[SEP]"] + a + ["[SEP]"]
        # segment_ids = [0] * (len(t) + len(q) + 2) + [1] * (len(a) + 2)

        segment_ids = []
        first_sep = True
        current_segment_id = 0
        for token in tokens:
            segment_ids.append(current_segment_id)
            if token == "[SEP]":
                if first_sep:
                    first_sep = False 
                else:
                    current_segment_id = 1

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        input_ids += ([0] * padding_length)

        input_mask += ([0] * padding_length)
        segment_ids += ([0] * padding_length)
        choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = example.label
        if example_index < 1 and is_training:
            logger.info("*** Example ***")
            logger.info("idx: {}".format(example_index))
            logger.info("guid: {}".format(example.guid))
            logger.info("tokens: {}".format(' '.join(tokens).replace('\u2581', '_')))
            logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
            logger.info("input_mask: {}".format(' '.join(map(str, input_mask))))
            logger.info("segment_ids: {}".format(' '.join(map(str, segment_ids))))
            logger.info("label: {}".format(label))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
                label=label
            )
        )
    return features


def _truncate_seq_pair(q, a, max_sequence_length, q_max_len=126, a_max_len=126):
    """Truncates a sequence pair in place to the maximum length."""

    q = q[:q_max_len]
    a = a[:a_max_len]

    return q, a


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=args.output_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
    #                                             num_training_steps=t_total)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=5 * len(train_dataloader))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    best_metric = 0
    patience = 0
    num_eva_epoch = 0
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        pgd = PGD(model)
        K = 3
        fgm = FGM(model)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert',
                                                                           'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if args.adversarial == "pgd":
                pgd.backup_grad()
                # 对抗训练
                for t in range(K):
                    pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
                    if t != K-1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    loss_adv = model(**inputs)[0]
                    loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                pgd.restore() # 恢复embedding参数

            elif args.adversarial == "fgm":
                # 对抗训练
                fgm.attack() # 在embedding上添加对抗扰动
                loss_adv = model(**inputs)[0]
                loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                fgm.restore() # 恢复embedding参数

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        num_eva_epoch += 1
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                    current_metric = results['metric']
                    if current_metric <= best_metric:
                        patience += 1
                        print("=" * 80)
                        print("Best Metric", best_metric)
                        print("Current Metric", current_metric)
                        print("=" * 80)
                        if patience > args.early_stop:
                            print("Out of patience !  Stop Training !")
                            return global_step, tr_loss / global_step
                    else:
                        patience = 0
                        print("=" * 80)
                        print("Best Metric", best_metric)
                        print("Current Metric", current_metric)
                        print("Saving Model......")
                        print("=" * 80)
                        best_metric = current_metric
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)

                    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                    with open(output_eval_file, "a") as writer:
                        logger.info("***** Eval results  *****")
                        writer.write("***** Eval results  *****\n")
                        writer.write("Num eva epoch = %s\n" % num_eva_epoch)
                        for key in sorted(results.keys()):
                            logger.info("  %s = %s", key, str(results[key]))
                            writer.write("%s = %s\n" % (key, str(results[key])))
                        writer.write("Best Metric = %s\n\n" % best_metric)

                # if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                #     # Save model checkpoint
                #     output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                #     if not os.path.exists(output_dir):
                #         os.makedirs(output_dir)
                #     model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                #     model_to_save.save_pretrained(output_dir)
                #     torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                #     logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    # eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_output_dir in eval_outputs_dirs:
        train_examples = read_examples(os.path.join(args.data_dir, 'dev.csv'), is_training=True)
        features = convert_examples_to_features(
            train_examples, tokenizer, args.max_seq_length, False)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
        eval_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_labels)
        # eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert',
                                                                               'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        print("preds: ", preds)
        print("out_label_ids: ", out_label_ids)
        # if args.output_mode == "classification":
        #     preds = np.argmax(preds, axis=1)
        # elif args.output_mode == "regression":
        #     preds = np.squeeze(preds)
        preds = np.argmax(preds, axis=1)
        # print(preds)
        # print(out_label_ids)
        # out_label_ids = out_label_ids[::2]
        result = compute_metrics(preds, out_label_ids)
        results.update(result)

    return results


def predict(args, model, tokenizer, prefix=""):
    np.set_printoptions(suppress=True)
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    predict_outputs_dirs = (args.predict_output_dir,)

    for eval_output_dir in predict_outputs_dirs:
        train_examples = read_examples(os.path.join(args.test_dir, 'test.csv'), is_training=False)
        features = convert_examples_to_features(
            train_examples, tokenizer, args.max_seq_length, True)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_labels)
        eval_dataset = train_data

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running prediction {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                          # XLM don't use segment_ids
                          'labels': batch[3]}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        # if args.output_mode == "classification":
        #     preds = np.argmax(preds, axis=1)
        # elif args.output_mode == "regression":
        #     preds = np.squeeze(preds)
        # print(preds)
        # preds = np.argmax(preds, axis=1)
        preds = np.array(preds)

        output_pred_file = os.path.join(eval_output_dir, "pred_results_" + prefix + ".jsonl")
        logger.info("***** Pred results {} *****".format(prefix))
        label_list = [0, 1]
        items = []
        for i in range(len(preds)):
            item = {}
            item['id'] = i
            print(len(preds[i]))
            item['labels'] = str(preds[i])
            items.append(item)
        write_file(items, output_pred_file)

    return preds


def main():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=False,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=False,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=False,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    # parser.add_argument("--task_name", default=None, type=str, required=True,
    #                     help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cuda_device", default=None, type=str, required=False,
                        help="cuda_device.")

    ## Other parameters
    parser.add_argument("--adversarial", default="", type=str,
                        help="Adversarial training method")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--early_stop", default=4, type=int,
                        help="early stop when f1 not increases any more")
    parser.add_argument("--num_labels", default=2, type=int,
                        help="num_labels")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=float,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout", default=0, type=float,
                        help="Linear dropout.")

    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run predict on the test set.")  # predict
    parser.add_argument("--test_dir", default=None, type=str, required=False,
                        help="The test data dir.")  # predict data dir
    parser.add_argument("--predict_model_dir", default=None, type=str, required=False,
                        help="predict_model_dir.")  # predict model dir
    parser.add_argument("--predict_output_dir", default=None, type=str, required=False,
                        help="predict_output_dir.")  # predict output dir
    parser.add_argument("--predict_model_num", default=-1, type=int,
                        help="predict_model_num.")  # predict output dir

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=args.num_labels,
                                          # finetuning_task=args.task_name,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_examples = read_examples(os.path.join(args.data_dir, 'train.csv'), is_training=True)
        features = convert_examples_to_features(
            train_examples, tokenizer, args.max_seq_length, True)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_labels)
        train_dataset = train_data
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    # Prediction
    if args.do_predict and args.local_rank in [-1, 0]:
        args.model_type = args.model_type.lower()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None)
        checkpoints = []
        for i in range(args.predict_model_num):
            model_path = os.path.join(args.predict_model_dir + str(i + 1), "checkpoint-best")
            checkpoints.append(model_path)
        for i in range(len(checkpoints)):
            checkpoint = checkpoints[i]
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            preds = predict(args, model, tokenizer, prefix=str(i + 1))
        return None


    return results


if __name__ == "__main__":
    main()
