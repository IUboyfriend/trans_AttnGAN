import torch
import torch.nn as nn

import numpy as np
from miscc.config import cfg

from GlobalAttention import func_attention
from transformers import BertTokenizer
import torch.nn.functional as F


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
    if class_ids is not None:
        for i in range(batch_size):
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        # masks = torch.ByteTensor(masks)
        masks = torch.BoolTensor(masks)  # Changed from torch.ByteTensor(masks)
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
    scores0 = scores0 / norm0.clamp(min=eps) * cfg.TRAIN.SMOOTH.GAMMA3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    if class_ids is not None:
        scores0.data.masked_fill_(masks, -float('inf'))
    scores1 = scores0.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1


def words_loss(img_features, words_emb, labels,
               cap_lens, class_ids, batch_size):
    """def sent_loss(cnn_code, rnn_code, labels, class_ids,
              batch_size, eps=1e-8):
    # ### Mask mis-match samples  ###
    # that come from the same class as the real sample ###
    masks = []
    if class_ids is not None:
        for i in range(batch_size):
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        # masks = torch.ByteTensor(masks)
        masks = torch.BoolTensor(masks)  # Changed from torch.ByteTensor(masks)
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
    scores0 = scores0 / norm0.clamp(min=eps) * cfg.TRAIN.SMOOTH.GAMMA3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    if class_ids is not None:
        scores0.data.masked_fill_(masks, -float('inf'))
    scores1 = scores0.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1

        words_emb(query): batch x nef x seq_len
        img_features(context): batch x nef x 17 x 17
    """
    masks = []
    att_maps = []
    similarities = []
    cap_lens = cap_lens.data.tolist()
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

        # masks = torch.ByteTensor(masks)
        masks = torch.BoolTensor(masks)  # Changed from torch.ByteTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda()

    similarities = similarities * cfg.TRAIN.SMOOTH.GAMMA3
    if class_ids is not None:
        similarities.data.masked_fill_(masks, -float('inf'))
    similarities1 = similarities.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1, att_maps


def sent_loss_bert(pred, target, labels, class_ids,
              batch_size, eps=1e-8):
    # ### Mask mis-match samples  ###
    # that come from the same class as the real sample ###
    if labels is not None:
        print(nn.MSELoss()(pred, target))
        print(nn.MSELoss()(pred, target).shape)
        return nn.MSELoss()(pred, target)


def words_loss_bert(pred, target, labels,
               cap_lens, class_ids, batch_size):
    """
        words_emb(query): batch x nef x seq_len
        img_features(context): batch x nef x 17 x 17
    """
    if labels is not None:
        # Predicted word embeddings and targeted word embeddings
        max_sent_len=max(pred.shape[2],target.shape[2])
        min_sent_len=min(pred.shape[2],target.shape[2])
        # Padding for Alignment
        pad = torch.zeros_like(torch.empty(pred.shape[0],pred.shape[1],max_sent_len - min_sent_len)).cuda()
        if pred.shape[2]<max_sent_len:
            pred = torch.cat((pred,pad),2)
        if target.shape[2]< max_sent_len:
            target = torch.cat((target,pad),2)
        return nn.MSELoss()(pred, target)

def sent_loss_bert_new(pred_sent_emb, sent_emb, labels, class_ids, batch_size):
    # Normalize the embeddings, L2 normalization to ensure the loss is direction-based but not magnitude-based
    pred_sent_norm = F.normalize(pred_sent_emb, p=2, dim=1)  # Shape: batch_size x emb_dim
    target_sent_norm = F.normalize(sent_emb, p=2, dim=1)     # Shape: batch_size x emb_dim

    # Expand dimensions to prepare for batch matrix multiplication
    # Shape after unsqueeze: batch_size x 1 x emb_dim
    pred_sent_norm = pred_sent_norm.unsqueeze(1)
    target_sent_norm = target_sent_norm.unsqueeze(2)  # Shape: batch_size x emb_dim x 1

    # Batch matrix multiplication to get cosine similarity
    # Shape after bmm: batch_size x 1 x 1
    cos_sim = torch.bmm(pred_sent_norm, target_sent_norm).squeeze()
    cos_sim = cos_sim.mean()
    sentence_loss = 1 - cos_sim
    # Loss: Negative log of cosine similarities

    if labels is not None:
        return sentence_loss

def words_loss_bert_new(pred, target, labels, cap_lens, class_ids, batch_size,pred_cap_len,real_cap_len):
    # Normalize the word embeddings

    pred_norm = F.normalize(pred, p=2, dim=2)  # Normalize along the embedding dimension
    target_norm = F.normalize(target, p=2, dim=2)

    # Creating masks based on actual lengths, boolean tensor
    pred_mask = torch.arange(pred.size(2), device=pred.device)[None, :].expand(batch_size, pred.size(2)) < pred_cap_len[:, None]
    target_mask = torch.arange(target.size(2),device=pred.device)[None, :].expand(batch_size, target.size(2)) < real_cap_len[:, None]

    # Apply masks to normalized embeddings
    pred_masked = pred_norm * pred_mask.unsqueeze(1).to(pred.device).float()
    target_masked = target_norm * target_mask.unsqueeze(1).to(target.device).float()
    #

    # Compute cosine similarity with masking
    cos_sim_matrix = torch.bmm(pred_masked.transpose(1, 2),
                               target_masked)  # shape: batch_size x pred_seq_len x target_seq_len
    #
    # print(cos_sim_matrix[-1,:,:])

    softmax_scores = torch.softmax(cos_sim_matrix, dim=1)

    scores_soft_max = torch.max(softmax_scores, dim=1).values

    # Apply the mask to the scores to consider only non-padding elements
    valid_scores = scores_soft_max * target_mask

    # print("non-padding soft-max score:",valid_scores.shape)
    # print(valid_scores)
    sum_scores = torch.sum(valid_scores, dim=1)
    num_non_padding = torch.sum(target_mask, dim=1)
    mean_scores = sum_scores / num_non_padding

    total_score = torch.sum(mean_scores)
    # print(total_score)
    word_loss = (batch_size - total_score)/batch_size
    # print(word_loss)
    return word_loss



# ##################Loss for G and Ds##############################
def discriminator_loss(netD, real_imgs, fake_imgs, conditions,
                       real_labels, fake_labels):
    # Forward
    real_features = netD(real_imgs)
    fake_features = netD(fake_imgs.detach())
    # loss
    #
    cond_real_logits = netD.COND_DNET(real_features, conditions)
    cond_real_errD = nn.BCELoss()(cond_real_logits, real_labels)
    cond_fake_logits = netD.COND_DNET(fake_features, conditions)
    cond_fake_errD = nn.BCELoss()(cond_fake_logits, fake_labels)
    #
    batch_size = real_features.size(0)
    cond_wrong_logits = netD.COND_DNET(real_features[:(batch_size - 1)], conditions[1:batch_size])
    cond_wrong_errD = nn.BCELoss()(cond_wrong_logits, fake_labels[1:batch_size])

    if netD.UNCOND_DNET is not None:
        real_logits = netD.UNCOND_DNET(real_features)
        fake_logits = netD.UNCOND_DNET(fake_features)
        real_errD = nn.BCELoss()(real_logits, real_labels)
        fake_errD = nn.BCELoss()(fake_logits, fake_labels)
        errD = ((real_errD + cond_real_errD) / 2. +
                (fake_errD + cond_fake_errD + cond_wrong_errD) / 3.)
    else:
        errD = cond_real_errD + (cond_fake_errD + cond_wrong_errD) / 2.
    return errD


def generator_loss(netsD, image_encoder, fake_imgs, real_labels,
                   words_embs, sent_emb, match_labels,
                   cap_lens, class_ids):
    numDs = len(netsD)
    batch_size = real_labels.size(0)
    logs = ''
    # Forward
    errG_total = 0
    for i in range(numDs):
        features = netsD[i](fake_imgs[i])
        cond_logits = netsD[i].COND_DNET(features, sent_emb)
        cond_errG = nn.BCELoss()(cond_logits, real_labels)
        if netsD[i].UNCOND_DNET is  not None:
            logits = netsD[i].UNCOND_DNET(features)
            errG = nn.BCELoss()(logits, real_labels)
            g_loss = errG + cond_errG
        else:
            g_loss = cond_errG
        errG_total += g_loss
        # err_img = errG_total.data[0]
        logs += 'g_loss%d: %.2f ' % (i, g_loss.item())

        # Ranking loss
        if i == (numDs - 1):
            # words_features: batch_size x nef x 17 x 17
            # sent_code: batch_size x nef
            region_features, cnn_code = image_encoder(fake_imgs[i])
            w_loss0, w_loss1, _ = words_loss(region_features, words_embs,
                                             match_labels, cap_lens,
                                             class_ids, batch_size)
            w_loss = (w_loss0 + w_loss1) * \
                cfg.TRAIN.SMOOTH.LAMBDA
            # err_words = err_words + w_loss.data[0]

            s_loss0, s_loss1 = sent_loss(cnn_code, sent_emb,
                                         match_labels, class_ids, batch_size)
            s_loss = (s_loss0 + s_loss1) * \
                cfg.TRAIN.SMOOTH.LAMBDA
            # err_sent = err_sent + s_loss.data[0]

            errG_total += w_loss + s_loss
            logs += 'w_loss: %.2f s_loss: %.2f ' % (w_loss.item(), s_loss.item())
    return errG_total, logs


def generator_loss_bert(netsD, image_caption, text_encoder, fake_imgs, real_labels,
                   words_embs, sent_emb, match_labels,
                   cap_lens, class_ids, real_cap_len):
    numDs = len(netsD)
    batch_size = real_labels.size(0)
    logs = ''
    # Forward
    errG_total = 0

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for i in range(numDs):
        features = netsD[i](fake_imgs[i])
        # Compute the conditional logits using the discriminator's conditional network
        cond_logits = netsD[i].COND_DNET(features, sent_emb)
        # Compute the conditional loss using binary cross-entropy loss
        cond_errG = nn.BCELoss()(cond_logits, real_labels)
        if netsD[i].UNCOND_DNET is not None:
            logits = netsD[i].UNCOND_DNET(features)
            errG = nn.BCELoss()(logits, real_labels)
            g_loss = errG + cond_errG
        else:
            g_loss = cond_errG
        errG_total += g_loss
        # err_img = errG_total.data[0]
        logs += 'g_loss%d: %.2f ' % (i, g_loss.item())

        # Innovation part: text-image matching loss
        # Modification, only for the last image
        if i == (numDs - 1):
            # words_features: batch_size x nef x 17 x 17
            # sent_code: batch_size x nef

            # Modification
            # embs = list(text_encoder.pipe(image_caption(fake_imgs[i])))
            #
            # pred_sent_emb = torch.Tensor(np.array([i.vector for i in embs])).cuda()
            # max_sent_len = max(1,len(max(embs,key=len)))
            # pred_words_embs=[]
            # for i in embs:
            #     pred_word_emb = [w.vector for w in i]
            #     sent_len =  len(i)
            #     if sent_len<max_sent_len:
            #         pred_word_emb+=[[0]*len(i[0].vector)]*(max_sent_len-sent_len)
            #     pred_words_embs.append(pred_word_emb)
            # pred_words_embs = torch.Tensor(np.array(pred_words_embs)).cuda()
            # pred_words_embs = pred_words_embs.permute(0,2,1)


            tokenized_inputs = tokenizer(
                image_caption(fake_imgs[i]),
                padding=True,  # Pad to the longest sequence in the batch
                truncation=True,  # Truncate to max length of the model
                return_tensors='pt'  # Return PyTorch tensors
            )
            # Forward pass through the BERT model
            with torch.no_grad():
                outputs = text_encoder(**tokenized_inputs)

            last_hidden_states = outputs.last_hidden_state



            # sentence feature vectors
            pred_sent_emb = last_hidden_states[:, 0, :].cuda()  # Shape: (batch_size, hidden_size)

            pred_words_embs = last_hidden_states[:, 1:,:].cuda()  # Shape: (batch_size, sequence_length, hidden_size = 768)

            pred_words_embs = pred_words_embs.permute(0, 2, 1)  # Shape: (batch_size, hidden_size = 768, sequence_length)


            attention_mask = tokenized_inputs['attention_mask'].cuda()
            pred_cap_len = attention_mask.sum(dim=1) - 2 # exclude the cls and the sep
            w_loss  = words_loss_bert_new(pred_words_embs, words_embs,
                                             match_labels, cap_lens,
                                             class_ids, batch_size,pred_cap_len,real_cap_len)
            w_loss = w_loss * \
                cfg.TRAIN.SMOOTH.LAMBDA
            # err_words = err_words + w_loss.data[0]

            # s_loss = sent_loss_bert(pred_sent_emb, sent_emb,
            #                              match_labels, class_ids, batch_size)

            #
            s_loss = sent_loss_bert_new(pred_sent_emb, sent_emb,
                                         match_labels, class_ids, batch_size)
            s_loss = s_loss * \
                cfg.TRAIN.SMOOTH.LAMBDA


            # err_sent = err_sent + s_loss.data[0]

            errG_total +=  cfg.TRAIN.SMOOTH.ALPHA * s_loss + cfg.TRAIN.SMOOTH.BETA * w_loss
            # print(cfg.TRAIN.SMOOTH.ALPHA * s_loss + cfg.TRAIN.SMOOTH.BETA * w_loss, errG_total)
            logs += 'w_loss: %.2f s_loss: %.2f ' % (w_loss.item(), s_loss.item())
    return errG_total, logs


##################################################################
def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD
