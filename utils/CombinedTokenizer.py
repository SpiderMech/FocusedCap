from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

import json
import os
from tqdm import tqdm

##############################
#        GLOBAL PARAMS       #
##############################
SEQ_START = "[CLS]"
SEQ_END = "[SEP]"


class FCTokenizer():
    def decode(self, token_ids, skip_special_tokens=False):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def encode(self, caption):
        return self.tokenizer.encode(caption)
    
    def get_vocab(self):
        return self.tokenizer.get_vocab()
    
    def __init__(self, train_dir) -> None:
        tokenizer_dir = f"./data/tokenizer_coco2014_vg.json"
        if os.path.exists(tokenizer_dir):
            self.tokenizer = Tokenizer.from_file(tokenizer_dir)
        else:
            self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            self.tokenizer.pre_tokenizer = Whitespace()
            trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", SEQ_START, SEQ_END], min_frequency=5)
            # self.tokenizer.train_from_iterator(all_captions, trainer)
            self.tokenizer.train([train_dir], trainer)
            self.tokenizer.post_processor = TemplateProcessing(
                single=f"{SEQ_START} $A {SEQ_END}",
                pair=f"{SEQ_START} $A {SEQ_END} $B:1 {SEQ_END}:1",
                special_tokens=[
                    (SEQ_START, self.tokenizer.token_to_id(SEQ_START)),
                    (SEQ_END, self.tokenizer.token_to_id(SEQ_END))
                ]
            )
            self.tokenizer.save(tokenizer_dir)
        self.vocab_size = self.tokenizer.get_vocab_size()

if __name__ == "__main__":
    tokenizer = FCTokenizer("../data/tokenizer_train.txt")
    output = tokenizer.encode("A pencil, A phone")
    print(output.ids)
    print(output.tokens)


##########################
#  EXTRACT BOTH CAPTIONS #
##########################

# print(f"Extrating training captions...")
# with open("./data/train_caption.json") as f:
#     training_caption = json.load(f)

# all_full_captions = [cap['caption'] for cap in training_caption]
# all_coco_ids = [cap['image_id'] for cap in training_caption]

# fc_d = {cap : 0 for cap in all_coco_ids}

# print(f"Extracting image meta data from visual genome")
# with open("./data/vg_image_data.json", "r") as f:
#     vg_image_data = json.load(f)

# # vg_to_coco = {}
# for d in vg_image_data:
#     if d['coco_id'] is not None:
#         coco_id = str(d['coco_id'])
#         if coco_id in fc_d:
#             fc_d[coco_id] = d['image_id']
# vg_to_coco = {val: key for key, val in fc_d.items()}
# print(f"{len(vg_to_coco)} coco images in vg dataset")
# # print(vg_to_coco)

# print(f"Extracting regional captions from visual genome")
# with open("./data/vg_region_descriptions.json", "r") as f:
#     vg_region_desc = json.load(f)

# print(f"Filtering visual genome regional captions")
# all_region_captions = []
# for d in tqdm(vg_region_desc):
#     regions = d['regions']
#     image_id = d['id']
#     if image_id in vg_to_coco:
#         for r in regions:
#             all_region_captions.append(r['phrase'])
# print(f"{len(all_region_captions)} regional captions collected")
# print(f"{len(all_full_captions) + len(all_region_captions)} captions in total")

# print("writing to file")
# with open("./data/tokenizer_train.txt", "w", encoding='utf-8') as f:
#     for caption in tqdm(all_full_captions):
#         f.write(caption + '\n')

#     for caption in tqdm(all_region_captions):
#         f.write(caption + '\n')
