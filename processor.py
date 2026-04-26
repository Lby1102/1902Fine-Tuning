from transformers import AutoProcessor
import config

def load_processor():
    processor=AutoProcessor.from_pretrained(config.model_dir)
    print('Processor loaded successfully')
    return processor

def process_batch(processor,images,labels):
    """
    To convert a batch of images and labels into tensors readable by the model
    """
    conversations=[]
    for image in images:
        conversation=[
            {
                "role":"user",
                "content":[
                    {"type":"image","image":image},
                    {"type":"text","text":"What are the characters in this verification code image?Please directly output the characters with no other any content"}
                ]
            }
        ]
        conversations.append(conversation)
    texts=[
        processor.apply_chat_template(conv ,tokenize=False,add_generattion_prompt=True)
        for conv in conversations
    ]
    inputs=processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True
    ).to(config.device)
    labels_encoded=processor.tokenizer(
        labels,
        return_tensors='pt',
        padding=True
    ).to(config.device)

    return inputs,labels_encoded
