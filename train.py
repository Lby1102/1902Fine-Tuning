import torch
from torch.optim import AdamW
import config
from dataset import build_dataloader
from processor import load_processor, process_batch
from model import load_model
import os

def train():
    train_loader,val_loader=build_dataloader(
        config.data_dir,
        config.batch_size,
        config.train_ratio
    )
    processor=load_processor()
    model=load_model()
    model.train()
    optimizer=AdamW(
        filter(lambda p:p.requires_grad,model.parameters()),
        lr=config.learning_rate
    )

    #training starts
    for epoch in range(config.num_epochs):
        total_loss=0
        num_batches=0
        for batch_idx,(images,labels) in enumerate(train_loader):
            inputs,labels_encoded=process_batch(processor,images,labels)
            outputs=model(
                **inputs,
                labels=labels_encoded['input_ids']
            )
            loss+=outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss+=loss.item()
            num_batches+=1

            if batch_idx % 100 == 0:
                avg_loss = total_loss / num_batches
                print(f"Batch {batch_idx}, 平均loss: {avg_loss:.4f}")
        "Evaluate on the validation set after each epoch."
        val_loss=evaluate(model,processor,val_loader)
        print(f"loss on validation data set: {val_loss:.4f}")

        save_path=os.path.join(config.output_dir,f"epoch_{epoch+1}")
        os.makedirs(save_path,exist_ok=True)
        model.save_pretrained(save_path)
        print(f"model has been saved into {save_path}")

def evaluate(model,processor,val_loader):
    '''computer loss on the validation data set'''
    model.eval()
    total_loss=0
    num_batches=0
    with torch.no_grad():
        for images,labels in val_loader:
            inputs, labels_encoded = process_batch(processor, images, labels)
            outputs=model(
                **inputs,
                labels=labels_encoded['input_ids']
            )
            total_loss+=outputs.loss.item()
            num_batches+=1
    model.train()
    return total_loss/num_batches

if __name__=='__main__':
    train()

