import os
import sys
import torch
import yaml

# 添加项目根目录到 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # 回到上一级目录
sys.path.insert(0, project_root)
from models.multimodal_encoder.t5_encoder import T5Embedder

GPU = 0
MODEL_PATH = "google/t5-v1_1-xxl"
CONFIG_PATH = "configs/base.yaml"

# Your instruction
INSTRUCTION = "Use the left hand to hook the book '皮囊' from the pile of books, then use the right hand to place it on the right bookshelf."

# Output path
OUTPUT_PATH = "data/baai/data/action176/action176.pt"

# Offload directory (if GPU VRAM < 24GB)
OFFLOAD_DIR = None

def main():
    with open(CONFIG_PATH, "r") as fp:
        config = yaml.safe_load(fp)
    
    device = torch.device(f"cuda:{GPU}")
    
    print(f"Loading T5 model from {MODEL_PATH}...")
    text_embedder = T5Embedder(
        from_pretrained=MODEL_PATH, 
        model_max_length=config["dataset"]["tokenizer_max_length"], 
        device=device,
        use_offload_folder=OFFLOAD_DIR
    )
    tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
    
    print(f"Encoding instruction: {INSTRUCTION}")
    
    # Tokenize
    tokenized_res = tokenizer(
        [INSTRUCTION], 
        return_tensors="pt",
        padding="longest",
        truncation=True
    )
    tokens = tokenized_res["input_ids"].to(device)
    attn_mask = tokenized_res["attention_mask"].to(device)
    
    # Encode
    with torch.no_grad():
        text_embeds = text_encoder(
            input_ids=tokens,
            attention_mask=attn_mask
        )["last_hidden_state"].detach().cpu()
    
    attn_mask = attn_mask.cpu().bool()
    
    # Extract only valid tokens (remove padding)
    text_embed = text_embeds[0][attn_mask[0]]
    
    # Save
    torch.save(text_embed, OUTPUT_PATH)
    print(f"✅ Saved language embedding to: {OUTPUT_PATH}")
    print(f"   Shape: {text_embed.shape}")

if __name__ == "__main__":
    main()