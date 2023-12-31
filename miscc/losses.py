import clip
import torch
import torch.nn as nn
from torch import autograd
from torch.nn import functional as F
from torchvision import transforms
import lpips
import math
import numpy as np
from miscc.config import cfg
from distributed import (
	get_rank,
	get_world_size,
)

from GlobalAttention import func_attention
from ipdb import set_trace

# ##################Loss for matching text-image###################
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
	"""Returns cosine similarity between x1 and x2, computed along dim.
	"""
	w12 = torch.sum(x1 * x2, dim)
	w1 = torch.norm(x1, 2, dim)
	w2 = torch.norm(x2, 2, dim)
	return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def sent_loss(cnn_code, rnn_code, labels, class_ids,
			  batch_size, eps=1e-8):
	# ### Mask mis-match samples  ###
	# that come from the same class as the real sample ###
	masks = []
	# logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
	# logit_scale = logit_scale.exp()
	if class_ids is not None:
		for i in range(batch_size):
			mask = (class_ids == class_ids[i]).astype(np.uint8)
			mask[i] = 0
			masks.append(mask.reshape((1, -1)))
		masks = np.concatenate(masks, 0)
		# masks: batch_size x batch_size
		masks = torch.ByteTensor(masks)
		if cfg.CUDA:
			masks = masks.cuda()

	# --> seq_len x batch_size x nef
	if cnn_code.dim() == 2:
		cnn_code = cnn_code.unsqueeze(0)
		rnn_code = rnn_code.unsqueeze(0)

	# cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
	cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
	rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
	# scores* / norm*: seq_len x batch_size x batch_size
	scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
	norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
	scores0 = scores0 / norm0.clamp(min=eps) * 100
	#scores0 = logit_scale*scores0 / norm0.clamp(min=eps)
	# --> batch_size x batch_size
	scores0 = scores0.squeeze()
	if class_ids is not None:
		scores0.data.masked_fill_(masks.bool(), -float('inf'))
	scores1 = scores0.transpose(0, 1)

	if labels is not None:
		loss0 = nn.CrossEntropyLoss()(scores0, labels)
		loss1 = nn.CrossEntropyLoss()(scores1, labels)
	else:
		loss0, loss1 = None, None
	return loss0, loss1



def words_loss_trainer_fine(img_features, words_emb, labels,
			   cap_lens, class_ids, batch_size):
	"""
		words_emb(query): batch x nef x seq_len
		img_features(context): batch x nef x 17 x 17
	"""
	masks = []
	att_maps = []
	similarities = []
	cap_lens = cap_lens
	for i in range(batch_size):
		if class_ids is not None:
			mask = (class_ids == class_ids[i]).astype(np.uint8)
			mask[i] = 0
			masks.append(mask.reshape((1, -1)))
		# Get the i-th text description
		words_num = cap_lens[i]
		# -> 1 x nef x words_num
		word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
		# -> batch_size x nef x words_num
		word = word.repeat(batch_size, 1, 1)
		# batch x nef x 17*17
		context = img_features
		"""
			word(query): batch x nef x words_num
			context: batch x nef x 17 x 17
			weiContext: batch x nef x words_num
			attn: batch x words_num x 17 x 17
		"""
		weiContext, attn = func_attention(word, context, cfg.TRAIN.SMOOTH.GAMMA1)
		att_maps.append(attn[i].unsqueeze(0).contiguous())
		# --> batch_size x words_num x nef
		word = word.transpose(1, 2).contiguous()
		weiContext = weiContext.transpose(1, 2).contiguous()
		# --> batch_size*words_num x nef
		word = word.view(batch_size * words_num, -1)
		weiContext = weiContext.view(batch_size * words_num, -1)
		#
		# -->batch_size*words_num
		row_sim = cosine_similarity(word, weiContext)
		# --> batch_size x words_num
		row_sim = row_sim.view(batch_size, words_num)

		# Eq. (10)
		row_sim.mul_(cfg.TRAIN.SMOOTH.GAMMA2).exp_()
		row_sim = row_sim.sum(dim=1, keepdim=True)
		row_sim = torch.log(row_sim)

		# --> 1 x batch_size
		# similarities(i, j): the similarity between the i-th image and the j-th text description
		similarities.append(row_sim)

	# batch_size x batch_size
	similarities = torch.cat(similarities, 1)
	if class_ids is not None:
		masks = np.concatenate(masks, 0)
		# masks: batch_size x batch_size
		masks = torch.ByteTensor(masks)
		if cfg.CUDA:
			masks = masks.cuda()

	similarities = similarities * cfg.TRAIN.SMOOTH.GAMMA3
	if class_ids is not None:
		similarities.data.masked_fill_(masks.bool(), -float('inf'))
	similarities1 = similarities.transpose(0, 1)
	if labels is not None:
		loss0 = nn.CrossEntropyLoss()(similarities, labels)
		loss1 = nn.CrossEntropyLoss()(similarities1, labels)
	else:
		loss0, loss1 = None, None
	return loss0, loss1, att_maps
# ##################Loss for G and Ds##############################
def discriminator_loss(netD, real_imgs, fake_imgs, conditions,
					   real_labels, fake_labels):
	# Forward
	real_features = netD(real_imgs)
	fake_features = netD(fake_imgs.detach())
	# loss
	#
	if len(cfg.GPU_ID) == 1:
		cond_real_logits = netD.COND_DNET(real_features, conditions)
		cond_fake_logits = netD.COND_DNET(fake_features, conditions)
	elif len(cfg.GPU_ID) > 1:
		cond_real_logits = netD.module.COND_DNET(real_features, conditions)
		cond_fake_logits = netD.module.COND_DNET(fake_features, conditions)
	cond_real_errD = nn.BCELoss()(cond_real_logits, real_labels)
	cond_fake_errD = nn.BCELoss()(cond_fake_logits, fake_labels)
	#
	batch_size = real_features.size(0)
	if len(cfg.GPU_ID) == 1:
		cond_wrong_logits = netD.COND_DNET(real_features[:(batch_size - 1)], conditions[1:batch_size])
	elif len(cfg.GPU_ID) > 1:
		cond_wrong_logits = netD.module.COND_DNET(real_features[:(batch_size - 1)], conditions[1:batch_size])
	cond_wrong_errD = nn.BCELoss()(cond_wrong_logits, fake_labels[1:batch_size])

	if len(cfg.GPU_ID) == 1:
		if netD.UNCOND_DNET is not None:
			real_logits = netD.UNCOND_DNET(real_features)
			fake_logits = netD.UNCOND_DNET(fake_features)
			real_errD = nn.BCELoss()(real_logits, real_labels)
			fake_errD = nn.BCELoss()(fake_logits, fake_labels)
			errD = ((real_errD + cond_real_errD) / 2. +
					(fake_errD + cond_fake_errD + cond_wrong_errD) / 3.)
		else:
			errD = cond_real_errD + (cond_fake_errD + cond_wrong_errD) / 2.
	
	elif len(cfg.GPU_ID) > 1:
		if netD.module.UNCOND_DNET is not None:
			real_logits = netD.module.UNCOND_DNET(real_features)
			fake_logits = netD.module.UNCOND_DNET(fake_features)
			real_errD = nn.BCELoss()(real_logits, real_labels)
			fake_errD = nn.BCELoss()(fake_logits, fake_labels)
			errD = ((real_errD + cond_real_errD) / 2. +
					(fake_errD + cond_fake_errD + cond_wrong_errD) / 3.)
		else:
			errD = cond_real_errD + (cond_fake_errD + cond_wrong_errD) / 2.

	log = 'Real_Acc: {:.4f} Fake_Acc: {:.4f} '.format(torch.mean(real_logits).item(), torch.mean(fake_logits).item())
	real_acc = torch.mean(real_logits).item()
	fake_acc = torch.mean(fake_logits).item()
	log = 'Real_Acc: {:.4f} Fake_Acc: {:.4f} '.format(real_acc, fake_acc)

	return errD, log, real_acc, fake_acc


def generator_loss(netD, image_encoder, fake_imgs, real_labels,
				   words_embs, sent_emb, match_labels,
				   cap_lens, class_ids):

	batch_size = real_labels.size(0)
	logs = ''
	G_loss = {}
	
	# Forward
	errG_total = 0

	features = netD(fake_imgs)
	if len(cfg.GPU_ID) == 1:
		cond_logits = netD.COND_DNET(features, sent_emb)
	elif len(cfg.GPU_ID) > 1:
		cond_logits = netD.module.COND_DNET(features, sent_emb)
	cond_errG = nn.BCELoss()(cond_logits, real_labels)
	
	if len(cfg.GPU_ID) == 1:
		if netD.UNCOND_DNET is not None:
			logits = netD.UNCOND_DNET(features)
			errG = nn.BCELoss()(logits, real_labels)
			g_loss = errG + cond_errG
		else:
			g_loss = cond_errG
	else:
		if netD.module.UNCOND_DNET is not None:
			logits = netD.module.UNCOND_DNET(features)
			errG = nn.BCELoss()(logits, real_labels)
			g_loss = errG + cond_errG
		else:
			g_loss = cond_errG

	errG_total += g_loss
	# err_img = errG_total.data[0]
	logs += 'errG: %.2f ' % (g_loss.item())
	G_loss['errG'] = '%.4f' % g_loss.item()

	# Ranking loss
	# words_features: batch_size x nef x 17 x 17
	# sent_code: batch_size x nef
	region_features, cnn_code = image_encoder(fake_imgs)
	w_loss0, w_loss1, _ = words_loss(region_features, words_embs,
									 match_labels, cap_lens,
									 class_ids, batch_size)
	w_loss = (w_loss0 + w_loss1) * cfg.TRAIN.SMOOTH.LAMBDA
	# err_words = err_words + w_loss.data[0]

	s_loss0, s_loss1 = sent_loss(cnn_code, sent_emb,
								 match_labels, class_ids, batch_size)
	s_loss = (s_loss0 + s_loss1) * cfg.TRAIN.SMOOTH.LAMBDA
	# err_sent = err_sent + s_loss.data[0]

	errG_total += w_loss + s_loss
	logs += 'w_loss: %.2f s_loss: %.2f ' % (w_loss.item(), s_loss.item())
	G_loss['w_loss'] = '%.4f' % w_loss.item()
	G_loss['s_loss'] = '%.4f' % s_loss.item()

	return errG_total, logs, G_loss


##################################################################
def KL_loss(mu, logvar):
	# -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
	KLD = torch.mean(KLD_element).mul_(-0.5)
	return KLD


##################################################################

def g_perception_loss(real_img, fake_imgs):
    perceptual_loss = torch.tensor(0.0, device=real_img.device)

    for this_fake_img in fake_imgs:
        this_real_img = F.adaptive_avg_pool2d(real_img, output_size=this_fake_img.shape[-2:])
        perceptual_loss += F.mse_loss(this_real_img, this_fake_img)
    
    return perceptual_loss


def d_logistic_loss(netD, real_img, fake_img, 
	c_code=None, real_labels=None, fake_labels=None):
	real_pred, cond_real_logits = netD(real_img, c_code)
	fake_pred, cond_fake_logits = netD(fake_img, c_code)
	real_loss = F.softplus(-real_pred)
	fake_loss = F.softplus(fake_pred)

	# todo cond_real_loss, cond_fake_loss, cond_wrong_loss

	d_loss = (real_loss.mean()  + fake_loss.mean())

	return d_loss, real_pred, fake_pred

	
def d_r1_loss(netD, real_img, c_code=None):
	real_pred, _ = netD(real_img)
	grad_real, = autograd.grad(
		outputs=real_pred.sum(), inputs=real_img, create_graph=True
	)
	grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
	
	return grad_penalty, real_pred


def g_nonsaturating_loss(netD, fake_img, c_code, real_labels):
	fake_pred, cond_logits = netD(fake_img, c_code)
	real_loss = F.softplus(-fake_pred).mean()

	# cond_logits = cond_net(fake_feat, c_code)
	cond_real_loss = nn.BCELoss()(cond_logits, real_labels)
	
	g_loss = real_loss + cond_real_loss

	return g_loss


def pixel_g_nonsaturating_loss(
	netD, real_img,fake_imgs, c_code, real_labels
):
	# pixel train方法最后没有用这个loss
	#perceptual_loss = torch.tensor(0.0, device=real_img.device)

	# for this_fake_img in fake_imgs:
	# 	this_real_img = F.adaptive_avg_pool2d(real_img, output_size=this_fake_img.shape[-2:])
	# 	perceptual_loss += F.mse_loss(this_real_img, this_fake_img)

	fake_pred, cond_logits = netD(fake_imgs, c_code)
	real_loss = F.softplus(-fake_pred).mean()

	# cond_logits = cond_net(fake_feat, c_code)
	#cond_real_loss = nn.BCELoss()(cond_logits, real_labels)
	
	g_loss = real_loss  

	return g_loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
	noise = torch.randn_like(fake_img) / math.sqrt(
		fake_img.shape[2] * fake_img.shape[3]
	)

	grad, = autograd.grad(
		outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
	)

	path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

	path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

	path_penalty = (path_lengths - path_mean).pow(2).mean()

	return path_penalty, path_mean.detach(), path_lengths
class LPIPS_LOSS():
	def __init__(self,use_gpu):
		spatial = True         # Return a spatial map of perceptual distance.
		self.loss_fn = lpips.LPIPS(net='alex', spatial=spatial) # Can also set net = 'squeeze' or 'vgg'
		if(use_gpu == 'cuda'):
			self.loss_fn.cuda()
	
	def lpips(self,real_img,fake_img):
		
		dist = self.loss_fn.forward(real_img, fake_img)
		result = torch.from_numpy(np.array(dist.mean().item())).cuda()
		return result

class CLIPLoss(torch.nn.Module):
	def __init__(self, model):
		super(CLIPLoss, self).__init__()
		# RN50 or ViT-B/32
		# self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
		self.model = model
		# self.upsample = torch.nn.Upsample(scale_factor=28)
		# self.avg_pool = torch.nn.AvgPool2d(kernel_size=32)
		self.preprocess = transforms.Resize([224, 224])

	def forward(self, image, text, labels):
		# image = self.avg_pool(self.upsample(image))
		# similarity = 1 - self.model(image, text)[0] / 100
		image = self.preprocess(image)
		logits_per_image, logits_per_text = self.model(image, text)
		loss = nn.CrossEntropyLoss()(logits_per_image, labels)
		return loss

