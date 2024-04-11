
from torch.utils.data import DataLoader

from transformers import BlipProcessor, BlipForConditionalGeneration



from miscc.config import cfg, cfg_from_file
from datasets import TextDataset, prepare_data
import torch.nn as nn
import os
import sys
import random

import numpy as np
import time

# Set the environment variable
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import torchvision.transforms as transforms

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)
from transformers import AutoProcessor, BlipForConditionalGeneration
from datasets import TextDataset  # Ensure this is your custom dataset module
import matplotlib.pyplot as plt


def show_images(images, num_images=4):
    fig, axs = plt.subplots(1, num_images, figsize=(15, 15))
    axs = axs.flatten()
    for img, ax in zip(images[:num_images], axs):
        img = img.transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()



class Blip_fineTuner:
    def __init__(self, data_loader, ixtoword,output_dir, num_epochs = 1): # 1e-6 trained for 260 batches

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model.to(self.device)

        self.data_loader = data_loader
        self.num_epochs = num_epochs
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-7)
        self.num_batches = len(self.data_loader)
        self.ixtoword = ixtoword
        self.output_dir = output_dir


    def train(self):
        fine_tuning_iterations = 0
        for epoch in range(self.num_epochs):
            start_t = time.time()
            step = 0
            while step < self.num_batches:
                batch_start_t = time.time()
                ######################################################
                # (1) Prepare training data, Generate the word and sentence embeddings
                ######################################################
                data = next(iter(self.data_loader))
                imgs, captions, cap_lens, class_ids, keys = prepare_data(data)
                # print(len(imgs)) # 3 scales, 64, 128, 256

                # show_images(imgs[0], num_images=4)  # Adjust 'num_images' as needed
                # if len(normalized_imgs) > 0:
                #     print(normalized_imgs[1].shape) # torch.Size([batch size, 3 RGB , scale = 64/128/256, scale])


                normalized_imgs = [(img+1.0) * 127.5 for img in imgs]
                normalized_imgs = [img.cpu().numpy().astype(np.uint8) for img in normalized_imgs]


                sentence_list = []
                for i in range(cfg.TRAIN.BATCH_SIZE):
                    cap = captions[i].data.cpu().numpy()
                    sentence = []
                    for j in range(len(cap)):
                        if cap[j] == 0:
                            break
                        word = self.ixtoword[cap[j]].encode('ascii', 'ignore').decode('ascii')
                        sentence.append(word)
                    sentence = " ".join(sentence)
                    sentence_list.append(sentence)

                inputs = self.processor(images=normalized_imgs[2], text=sentence_list, padding = 'longest' , return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # for key, value in inputs.items():


                #     print(f"Shape of {key}: {value.shape}")
                #     # Print a small part of the tensor value, e.g., first few elements
                #     print(f"Value of {key} (partial view): {value[0, :]}")  # adjust the slicing as needed
                #     print()  # for an empty line between different keys

                outputs = self.model(input_ids = inputs['input_ids'], pixel_values = inputs['pixel_values'], labels =inputs['input_ids'] , attention_mask=inputs['attention_mask'])

                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                batch_end_t = time.time()
                if (fine_tuning_iterations + 1) % 20 == 0:
                    checkpoint_path = os.path.join(self.output_dir, f"_BLIP_epoch_{epoch + 1}_batch_{step + 1}.pth")
                    torch.save({
                        'epoch': epoch,
                        'batch': step + 1,
                        'model_state_dict': self.model.state_dict(),
                        'loss': loss,
                    }, checkpoint_path)
                    print(f"Model saved !")
                batch_time = batch_end_t - batch_start_t
                print(f"Epoch {epoch + 1}, Step {step + 1}/{len(self.data_loader)}, Batch Time: {batch_time:.4f} seconds, Loss: {loss.item()}")
                # torch.cuda.empty_cache()
                step += 1
                fine_tuning_iterations +=1




    def getCaption(self, imgs):
        random.seed(120)
        np.random.seed(120)
        torch.manual_seed(120)
        if cfg.CUDA:
            torch.cuda.manual_seed_all(130)

        # Process and prepare the batch of images
        data = next(iter(self.data_loader))
        imgs, captions, cap_lens, class_ids, keys = prepare_data(data)
        # print("Fetched examples:", keys)
        # normalized_imgs = [(img + 1.0) / 2.0 for img in imgs]

        normalized_imgs = [(img + 1.0) * 127.5 for img in imgs]
        normalized_imgs = [img.cpu().numpy().astype(np.uint8) for img in normalized_imgs]


        inputs = self.processor(images=normalized_imgs[2], return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Original captions
        sentence_list = []
        for i in range(cfg.TRAIN.BATCH_SIZE):
            cap = captions[i].data.cpu().numpy()
            sentence = []
            for j in range(len(cap)):
                if cap[j] == 0:
                    break
                word = self.ixtoword[cap[j]].encode('ascii', 'ignore').decode('ascii')
                sentence.append(word)
            sentence = " ".join(sentence)
            sentence_list.append(sentence)

        # Generate captions for the batch
        ret_list = []
        outputs = self.model.generate(**inputs,  max_length=18, min_length=5, num_beams = 3)#num_beams=5,
        for output in outputs:
            caption = self.processor.decode(output, skip_special_tokens=True)
            ret_list.append(caption)
        #
        # show_images(imgs[2], num_images=4)
        show_images(normalized_imgs[2], num_images=5)
        return ret_list, sentence_list
if __name__ == '__main__':

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])

    dataset = TextDataset("../data/birds", 'train',
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)

    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size= cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))  #

    output_directory = "../output/BLIP"
    fine_tuner = Blip_fineTuner(data_loader=dataloader, ixtoword=dataset.ixtoword, output_dir=output_directory)

    # checkpoint = torch.load("D:/Study/fourthYear_second/FYP/using detectron 2/output/BLIP/_BLIP_epoch_1_batch_160.pth")
    # fine_tuner.model.load_state_dict(checkpoint['model_state_dict'])
    fine_tuner.train()

    #Load the fine-tuned model checkpoint
    # checkpoint = torch.load("../output/BLIP_Result/lr=1e-6, more epochs/_BLIP_epoch_1_batch_400.pth")
    # # print(checkpoint['model_state_dict'])
    # fine_tuner.model.load_state_dict(checkpoint['model_state_dict'])
    # captions, true_captions = fine_tuner.getCaption(dataloader)
    # print("Generated captions:\n" )
    # print(captions)
    # print("Real captions:\n" )
    # print( true_captions)
    # # print(checkpoint['loss'])

