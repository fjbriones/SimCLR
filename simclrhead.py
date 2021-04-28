import logging
import os
import sys

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint

import torchvision

torch.manual_seed(0)

class Head(nn.Module):

	def __init__(self, num_classes, in_dims=128):
		super(Head, self).__init__()
		self.fc1 = nn.Linear(in_dims, num_classes)

	def forward(self, x):
		x = self.fc1(x)
		return x

class SimCLRHead(object):

	def __init__(self, *args, **kwargs):
		self.args = kwargs['args']
		self.model = kwargs['model'].to(self.args.device)
		self.writer = SummaryWriter()
		logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training_head.log'), level=logging.DEBUG)
		

		for l in self.model.parameters():
			l.requires_grad = False

		self.head = Head(self.args.num_classes).to(self.args.device)
		self.optimizer = torch.optim.Adam(self.head.parameters(), self.args.lr, weight_decay=self.args.weight_decay)
		self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

	def train(self, train_loader, val_loader):
		save_config_file(self.writer.log_dir, self.args)

		n_iter = 0		

		logging.info(f"Start SimCLR head training for {self.args.epochs} epochs.")
		logging.info(f"Training with gpu: {not self.args.disable_cuda}.")

		max_top1_accuracy = 0
		max_top5_accuracy = 0

		for epoch_counter in range(self.args.epochs):
			top1_train_accuracy = 0
			for counter, (x_batch, y_batch) in tqdm(enumerate(train_loader)):
				images = x_batch.to(self.args.device)
				labels = y_batch.to(self.args.device)

				self.optimizer.zero_grad()

				hidden = self.model(images)
				logits = self.head(hidden)
				loss = self.criterion(logits, labels)
				top1, top5 = accuracy(logits, labels, topk=(1,5))
				top1_train_accuracy += top1[0]
				
				loss.backward()
				self.optimizer.step()

				if n_iter % self.args.log_every_n_steps == 0:
					self.writer.add_scalar('head_loss', loss, global_step=n_iter)
					self.writer.add_scalar('head_acc/top1', top1[0], global_step=n_iter)
					self.writer.add_scalar('head_acc/top5', top5[0], global_step=n_iter)

				n_iter += 1

			top1_train_accuracy /= (counter + 1)
			logging.debug(f"Training Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1_train_accuracy}")

			top1_accuracy = 0
			top5_accuracy = 0
			for counter, (x_batch, y_batch) in tqdm(enumerate(val_loader)):
				images = x_batch.to(self.args.device)
				labels = y_batch.to(self.args.device)

				hidden = self.model(x_batch)
				logits = self.head(hidden)

				top1, top5 = accuracy(logits, labels, topk=(1,5))
				top1_accuracy += top1[0]
				top5_accuracy += top5[0]

				top1_accuracy /= (counter + 1)
				top5_accuracy /= (counter + 1)

			top1_accuracy /= (counter + 1)
			top5_accuracy /= (counter + 1)
			logging.debug(f"Validation Epoch: {epoch_counter}\tTop1_accuracy: {top1_accuracy}\tTop5 accuracy: {top5_accuracy}")

			checkpoint_name = 'checkpoint_head_{:04d}.pth.tar'.format(epoch_counter)
			if max_top1_accuracy < top1_accuracy and max_top5_accuracy < top5_accuracy:
				save_checkpoint({
					'epoch': self.args.epoch,
					'arch': self.args.arch,
					'state_dict': self.model.state_dict(),
					'head_state_dict': self.head.state_dict(),
					'optimizer': self.optimizer.state_dict()
					}, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))

			# print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")

