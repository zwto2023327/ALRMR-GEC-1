import os
import re
import sys
from collections import defaultdict
from functools import partial

import numpy as np
import torch
from tqdm.auto import tqdm
import copy
import random
import gc

def attach_index(path, index, suffix=""):
    if re.search(suffix + "$", path):
        prefix, suffix = re.match(f"^(.*)({suffix})$", path).groups()
    else:
        prefix, suffix = path, ""
    return f"{prefix}_{index}{suffix}"


def get_batch_metrics(pred_labels, labels, mask=None, ignore_labels=None, metric_func=None,
                      threshold=0.5):
    answer = defaultdict(int)
    error_list = copy.deepcopy(pred_labels)
    for r, (curr_pred_labels, curr_labels) in enumerate(zip(pred_labels, labels)):
        if mask is not None:
            curr_labels = [x for x, flag in zip(curr_labels, mask[r]) if flag]
        elif ignore_labels is not None:
            curr_labels = [label for label in curr_labels if label not in ignore_labels]
        # assert len(curr_pred_labels) == len(curr_labels), f"{len(curr_pred_labels)}-{len(curr_labels)}"
        for key, value in metric_func(curr_labels, curr_pred_labels, threshold=threshold).items():
            if key != "error_list" and key != "TP_list" and key != "FN_list" and key != "FP_list":
                answer[key] += value
            else:
                if key == "error_list":
                    error_list[r] = value
    return {"answer": answer, "error_list": error_list}


def update_metrics(metrics, batch_output, batch, mask=None,
                   answer_field="labels", y_field="y", extract_func=None,
                   metric_func=None, aggregate_func=None, threshold=0.5):
    n_batches = metrics["n_batches"]
    for key, value in batch_output.items():
        if "loss" in key:
            metrics[key] = (metrics.get(key, 0.0) * n_batches + value.item()) / (n_batches + 1)
    metrics["n_batches"] += 1
    if extract_func is not None:
        y_pred, y_true = extract_func(batch_output, batch)
    else:
        y_pred, y_true = batch_output[answer_field], batch[y_field].cpu().tolist()
    batch_metrics = get_batch_metrics(y_pred, y_true, mask=mask, ignore_labels=None, metric_func=metric_func, threshold=threshold)

    for key, value in batch_metrics["answer"].items():
        metrics[key] = metrics.get(key, 0) + value
    # print(metrics)
    aggregate_func(metrics)
    return batch_metrics


def replace_index_note(index_map, index_list, elem):
    if elem.shape[0] == 0 or elem.shape[1] != 2:
        return elem
    delete = []
    index = -1
    for e in elem:
        index = index + 1
        if e[0] not in index_list or e[1] not in index_list:
            delete.append(index)
    elem = np.delete(elem, delete, axis=0)
    for key in index_map:
        elem[elem == key] = index_map[key]
    return elem

def get_batch_note( notelist, batch_note, correctlist, model, flag=True, mode="train"):
    # index_list永远是一维
    index_list = []
    index_map = {}
    index_add = 0
    sum = 0
    default = []
    default_i = 0
    default_index = []
    offset = [0]
    last_index_add = -1
    for index in range(len(notelist)):
        di = index - sum
        if di == batch_note["default"][default_i]:
            sum = sum + di + 1
            default_value = index_add - last_index_add - 1
            if default_value != 0:
                default_index.append(default_i)
                index_map[index] = index_add
                default.append(default_value)
                last_index_add = index_add
                index_add = index_add + 1
                offset.append(index_add)
                index_list.append(index)
            default_i = default_i + 1
            continue
        if index in correctlist and mode == "train" and flag == True:
            continue
        index_map[index] = index_add
        index_add = index_add + 1
        index_list.append(index)
    batch_note['input_ids'] = torch.index_select(batch_note['input_ids'], 0, torch.tensor(index_list).to(model.device))
    batch_note['words'] = [batch_note['words'][i] for i in index_list]
    batch_note['label'] = torch.index_select(batch_note['label'], 0, torch.tensor(index_list).to(model.device))
    batch_note['start'] = [batch_note['start'][i] for i in index_list]
    batch_note['end'] = [batch_note['end'][i] for i in index_list]
    batch_note['origin_start'] = [batch_note['origin_start'][i] for i in index_list]
    batch_note['origin_end'] = [batch_note['origin_end'][i] for i in index_list]
    batch_note['flag'] = [batch_note['flag'][i] for i in index_list]
    batch_note['target'] = [batch_note['target'][i] for i in index_list]
    batch_note['default'] = default
    batch_note['indexes'] = [batch_note['indexes'][i] for i in default_index]
    batch_note['hard_pairs'] = replace_index_note(index_map, index_list, batch_note['hard_pairs'])
    batch_note['soft_pairs'] = replace_index_note(index_map, index_list, batch_note['soft_pairs'])
    batch_note['no_change_pairs'] = replace_index_note(index_map, index_list, batch_note['no_change_pairs'])
    batch_note['offset'] = offset
    return {"batch_note": batch_note, "index_list": index_list}

class ModelTrainer:
    
    def __init__(self, epochs=1, initial_epoch=0,
                 checkpoint_dir=None, checkpoint_name="checkpoint.pt", save_all_checkpoints=False,
                 eval_steps=None, evaluate_after=False, validate_metric="accuracy", less_is_better=False):
        self.epochs = epochs
        self.initial_epoch = initial_epoch
        self.initmodelflag = True
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        else:
            self.checkpoint_path = None
        self.save_all_checkpoints = save_all_checkpoints
        self.eval_steps = eval_steps
        self.evaluate_after = evaluate_after
        self.validate_metric = validate_metric
        self.less_is_better = less_is_better
        self.notelist = {}
        self.renum = 0
        self.correctflag = 0
        self.flagnum = -1
        self.nochangenum = 0
        self.lasttrainacc = 0
        self.lastvalacc = 0
        self.correct_num = 0
        self.all_num = 0
        self.testvalidate = 0
        self.newflag = False
        self.correctflag = 0
        self.correctlist = {}
        self.min_lr = -1
        self.max_lr = -1

    def do_epoch(self, model, dataloader, mode="validate", epoch=0, eval_steps=None,
                 answer_field="labels", y_field="y",
                 extract_func=None, metric_func=None, aggregate_func=None, display_func=None,
                 ncols=200, dynamic_ncols=False, count_mode="batch", total=None,
                 check_field="input_ids", check_dim=1, max_length=512, **kwargs):
        metrics = {"n_batches": 0, "loss": 0.0}
        self.correct_num = 0
        self.all_num = 0
        func = model.train_on_batch if mode == "train" else model.validate_on_batch
        if count_mode == "batch":
            total = getattr(dataloader, "__len__", None)
        progress_bar = tqdm(total=total, leave=True, ncols=ncols, dynamic_ncols=dynamic_ncols)
        progress_bar.set_description(f"{mode}, epoch={(epoch + 1) if mode == 'train' else epoch}")
        evaluation_step = 0
        with progress_bar:
            for batch in dataloader:
                batch_len = len(batch['indexes'])
                prev_evaluation_step = evaluation_step
                if (mode == "train" and check_field in batch and batch[check_field].shape[check_dim] > max_length):
                    batch_metrics = dict()
                else:
                    batch_answers, mask = batch[y_field], batch.get("mask")
                    if mask is not None:
                        mask = mask.bool()
                    try:
                        if progress_bar.n <= -1 and mode == "train":
                            batch_metrics = dict()
                        else:
                            index_list = {}
                            noteindex = ""
                            for i in range(len(batch["indexes"])):
                                noteindex = noteindex + str(batch["indexes"][i]) + str(batch["start"][i]) + str(
                                    len(batch["label"]))
                            if (mode == "train" or self.testvalidate == 1) and noteindex in self.notelist:
                                batch_note = batch.copy()
                                if noteindex in self.correctlist and "error_list" in self.correctlist[noteindex]:
                                    batch_list = get_batch_note(self.notelist[noteindex]["error_list" ], batch_note, self.correctlist[noteindex]["error_list"],
                                                                        model=model, flag=True, mode=mode)
                                else:
                                    batch_list = get_batch_note(self.notelist[noteindex]["error_list" ], batch_note, correctlist={}, model=model, flag=False, mode=mode)
                                batch = batch_list["batch_note"]
                                index_list = batch_list["index_list"]
                                if len(batch["default"]) == 0 or len(index_list) == 0:
                                    continue
                            if self.newflag == True:
                                batch_output = func(batch, mask=mask, new_lr=self.new_lr, new_flag=self.newflag)
                                self.newflag = False
                            else:
                                batch_output = func(batch, mask=mask)

                            if mode == "train" or self.testvalidate == 1:
                                batch_metrics = update_metrics(
                                    metrics, batch_output, batch, mask, answer_field=answer_field, y_field=y_field,
                                    extract_func=extract_func, metric_func=metric_func, aggregate_func=aggregate_func)
                                if noteindex in self.notelist and self.initmodelflag == False:
                                    fir = len(batch_metrics["error_list"])
                                    num = 0
                                    for i in range(fir):
                                        sec = len(batch_metrics["error_list"][i])
                                        if sec != 0:
                                            for index in range(sec):
                                                if self.correctflag > 0 :
                                                    if self.testvalidate == 1:
                                                        if noteindex in self.correctlist and "error_list" in self.correctlist[noteindex] and index_list[num] in self.correctlist[noteindex]["error_list"] :
                                                            if batch_metrics["error_list"][i][index] == 1 :
                                                                self.correct_num = self.correct_num + 1
                                                            self.all_num = self.all_num + 1
                                                else:
                                                    if self.notelist[noteindex]["error_list"][index_list[num]] == 0 and batch_metrics["error_list"][i][index] == 1 and (num + 1) not in batch["offset"]:
                                                        if noteindex not in self.correctlist:
                                                            self.correctlist[noteindex] = {}
                                                        if "error_list" not in self.correctlist[noteindex]:
                                                            self.correctlist[noteindex]["error_list"] = {}
                                                        self.correctlist[noteindex]["error_list"][index_list[num]] = 1
                                                self.notelist[noteindex]["error_list"][index_list[num]] = batch_metrics["error_list"][i][index]
                                                num = num + 1
                                else:
                                    self.notelist[noteindex] = {}
                                    self.notelist[noteindex]["error_list"] = []
                                    fir = len(batch_metrics["error_list"])
                                    num = 0
                                    for i in range(fir):
                                        sec = len(batch_metrics["error_list"][i])
                                        if sec != 0:
                                            for index in range(sec):
                                                self.notelist[noteindex]["error_list"].append(batch_metrics["error_list"][i][index])
                                                num = num + 1
                                if count_mode == "sample":
                                    batch_size = batch_len
                            else:
                                batch_metrics = update_metrics(
                                    metrics, batch_output, batch, mask, answer_field=answer_field, y_field=y_field,
                                    extract_func=extract_func, metric_func=metric_func, aggregate_func=aggregate_func
                                )
                                if count_mode == "sample":
                                    batch_size = batch_metrics[
                                        "seq_total"] if "seq_total" in batch_metrics else batch_len

                    except ValueError:
                        continue
                postfix = display_func(metrics)
                if noteindex not in self.notelist and mode == "validate":
                    self.notelist[noteindex] = {}
                    self.notelist[noteindex]["error_list"] = []
                    fir = len(batch_metrics["error_list"])
                    num = 0
                    for i in range(fir):
                        sec = len(batch_metrics["error_list"][i])
                        if sec != 0:
                            for index in range(sec):
                                self.notelist[noteindex]["error_list"].append(batch_metrics["error_list"][i][index])
                                num = num + 1
                if mode == "validate" or self.testvalidate == 1:
                    fir = len(batch_metrics["error_list"])
                    num = 0
                    for i in range(fir):
                        sec = len(batch_metrics["error_list"][i])
                        if sec != 0:
                            for index in range(sec):
                                self.notelist[noteindex]["error_list"][num] = batch_metrics["error_list"][i][index]
                                num = num + 1
                progress_bar.update(batch_size if count_mode == "sample" else 1)
                postfix["lr"] = f"{model.scheduler.get_last_lr()[0]:.2e}"
                progress_bar.set_postfix(postfix)
                if mode == "train" and eval_steps is not None:
                    evaluation_step = progress_bar.n // eval_steps
                    if evaluation_step != prev_evaluation_step:
                        self.eval_func(model, epoch=f"{epoch}_{progress_bar.n}")
        return metrics
    
    def train(self, model, train_data, dev_data=None, total=None, dev_total=None, count_mode="sample", **kwargs):
        self.best_score = np.inf if self.less_is_better else -np.inf
        eval_steps = self.eval_steps if dev_data is not None else None
        self.eval_func = partial(
            self.evaluate_and_save_model, dev_data=dev_data, total=dev_total, count_mode=count_mode, **kwargs
        )
        self.new_lr = kwargs["lr"]
        file_path = "/home/boot/STU/workspaces/wzx/VSR/error_epoch.txt"
        if os.path.exists(file_path):
            os.remove(file_path)
        file = open(file_path, "a")
        dev_metrics = self.eval_func(model, epoch=self.initial_epoch)
        self.initmodel = True
        for epoch in range(self.initial_epoch, self.epochs):
            self.reverse = 1
            self.reverse_enable = True
            if self.correctflag == 0 and self.initmodel == True and self.newflag == True:
                path_to_load = attach_index(self.checkpoint_path, self.initial_epoch, "\.pt")
                model.load_state_dict(torch.load(path_to_load), False)
                del self.notelist
                gc.collect()
                self.notelist = {}
                file.write("===================init=======================\n")
                file.flush()
                self.initmodelflag = True
                train_metrics = self.do_epoch(
                    model, train_data, mode="train", epoch=self.initial_epoch, total=total,
                    eval_steps=eval_steps, count_mode=count_mode, **kwargs
                )

            else:
                train_metrics = self.do_epoch(
                    model, train_data, mode="train", epoch=epoch, total=total,
                    eval_steps=eval_steps, count_mode=count_mode, **kwargs
                )
            self.initmodelflag = False
            if len(self.correctlist) > 0 and self.correctflag == 0:
                self.correctflag = 1
            dev_metrics = self.eval_func(model, epoch=epoch + 1)
            self.trainacc = train_metrics["accuracy"]
            self.valacc = dev_metrics["accuracy"]
            if self.correctflag == 1:
                self.testvalidate = 1
                train_metrics = self.do_epoch(
                    model, train_data, mode="validate", epoch=epoch, total=total,
                    eval_steps=eval_steps, count_mode=count_mode, **kwargs
                )
                self.testvalidate = 0
                if self.all_num == 0:
                    file.write("The all_num is zero,skip now\n")
                    del self.correctlist
                    gc.collect()
                    self.correctlist = {}
                    self.correctflag = 0
                    file.flush()
                    continue
                self.correctflag = 2
                file.write("The epoch is {:.1f}\n".format(epoch))
                file.write("The trainacc is {:.3f}\n".format(self.trainacc))
                file.write("The valacc is {:.3f}\n".format(self.valacc))
                self.lastmero = self.correct_num / self.all_num
                self.stagenum = self.lastmero
                file.write("The number is {:.4f}\n".format(self.lastmero))
                file.write("The lr is {:.4e}\n".format(self.new_lr))
                file.write("\n")
                file.flush()
            elif self.correctflag == 2:
                lr = self.new_lr
                self.testvalidate = 1
                train_metrics = self.do_epoch(
                    model, train_data, mode="validate", epoch=epoch, total=total,
                    eval_steps=eval_steps, count_mode=count_mode, **kwargs
                )
                self.testvalidate = 0
                if self.all_num == 0:
                    file.write("The all_num is zero,skip now\n")
                    del self.correctlist
                    gc.collect()
                    self.correctlist = {}
                    self.correctflag = 0
                    file.flush()
                    continue
                self.nowmero = self.correct_num / self.all_num
                file.write("The number is {:.4f}\n".format(self.nowmero))
                if self.nowmero > self.lastmero + 0.001:
                    if self.flagnum == -1:
                        self.renum = 1
                    elif self.flagnum == 0:
                        self.renum = self.renum + 1
                    else:
                        self.renum = 0
                    self.flagnum = 0
                elif self.nowmero < self.lastmero - 0.001:
                    if self.flagnum == -1:
                        self.renum = 1
                    elif self.flagnum == 1:
                        self.renum = self.renum + 1
                    else:
                        self.renum = 0
                    self.flagnum = 1
                if self.initmodel == True:
                    if (self.nowmero > self.lastmero + 0.005) or (self.renum > 1 and self.nowmero > self.lastmero):
                        self.min_lr = self.new_lr
                        if self.max_lr > -1 and self.max_lr > self.new_lr:
                            self.new_lr = self.new_lr + (1 / 2) * (self.max_lr - self.new_lr)
                        else:
                            self.new_lr = self.new_lr * 10
                        self.newflag = True
                    elif (self.nowmero < self.lastmero - 0.005) or (self.renum > 1 and self.nowmero < self.lastmero):
                        self.max_lr = self.new_lr
                        if self.min_lr > -1 and self.min_lr < self.new_lr:
                            self.new_lr = self.new_lr - (1 / 2) * (self.new_lr - self.min_lr)
                        else:
                            self.new_lr = self.new_lr * (1 / 10)
                        self.newflag = True
                    else:
                        self.correctflag = 2
                        self.initmodel = False
                        self.nochangenum = self.nochangenum + 1
                else:
                    if (self.nowmero > self.stagenum + 0.01) or (self.renum > 1 and self.nowmero > self.lastmero):
                        self.new_lr = self.new_lr*1.2
                        self.newflag = True
                    elif (self.nowmero < self.stagenum - 0.01) or (self.renum > 1 and self.nowmero < self.lastmero):
                        self.new_lr = self.new_lr*0.8
                        self.newflag = True
                    else:
                        self.correctflag = 2
                        self.nochangenum = self.nochangenum + 1
                self.lastmero = self.nowmero
                if self.nochangenum > 5:
                    self.new_lr = self.new_lr * 0.9
                    self.newflag = True
                if self.newflag == True:
                    del self.correctlist
                    del self.notelist
                    gc.collect()
                    self.renum = 0
                    self.correctflag = 0
                    self.flagnum = -1
                    self.nochangenum = 0
                    self.initmodelflag = True
                    self.notelist = {}
                    self.correctlist = {}
                file.write("The epoch is {:.1f}\n".format(epoch))
                file.write("The trainacc is {:.3f}\n".format(self.trainacc))
                file.write("The valacc is {:.3f}\n".format(self.valacc))
                file.write("The lr is {:.4e}\n".format(self.new_lr))
                file.write("\n")
                file.flush()
            else:
                file.write("The epoch is {:.4f}\n".format(epoch))
                file.write("The xtrainacc is {:.3f}\n".format(self.trainacc))
                file.write("The xvalacc is {:.3f}\n".format(self.valacc))
        file.close()
        if dev_data is not None and self.evaluate_after:
            if self.checkpoint_path is not None and not self.save_all_checkpoints:
                model.load_state_dict(torch.load(self.checkpoint_path))
            self.do_epoch(model, dev_data, mode="validate", epoch="evaluate",
                          total=dev_total, count_mode=count_mode, **kwargs)
        return

    def is_better_score(self, epoch_score, best_score):
        if epoch_score is None:
            return False
        return (self.less_is_better == (epoch_score <= best_score))
    
    def evaluate_and_save_model(self, model, dev_data, epoch=None, total=None, **kwargs):
        if dev_data is not None:
            dev_metrics = self.do_epoch(model, dev_data, mode="validate", epoch=epoch, total=total, **kwargs)
            epoch_score = dev_metrics.get(self.validate_metric)
            to_save_checkpoint = self.save_all_checkpoints
            if self.is_better_score(epoch_score, self.best_score):
                to_save_checkpoint, self.best_score = True, epoch_score
        else:
            dev_metrics, to_save_checkpoint = None, True
        if to_save_checkpoint and self.checkpoint_path is not None:
            path_to_save = (attach_index(self.checkpoint_path, epoch, "\.pt") if self.save_all_checkpoints
                            else self.checkpoint_path)
            torch.save(model.state_dict(), path_to_save)
        return dev_metrics