from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer

def get_test_prompts() -> list[str]:
    return [
        # "The capital of France is",
        # "Machine learning is",
        "Python is a programming language that",
        "In the year 2024",
        "Artificial intelligence will be",
        "The weather today is",
        # "Scientists have discovered",
        # "The most important thing in life is",
    ]


class PromptDataset(Dataset):
    def __init__(self, prompts: list[str], tokenizer, device: str = "cuda"):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.device = device
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        encoded = self.tokenizer(prompt, return_tensors="pt")
        
        input_ids = encoded.input_ids.squeeze(0)  # Remove batch dimension
        attention_mask = encoded.attention_mask.squeeze(0)  # Remove batch dimension
        position_ids = torch.arange(
            input_ids.shape[0], dtype=torch.long
        )
        
        return {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'attention_mask': attention_mask
        }

prompts = get_test_prompts()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
device = "cuda"
dataset = PromptDataset(prompts, tokenizer, device)
    
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,  # Keep original order
#    collate_fn=collate_fn
)
breakpoint()
data_iter = iter(dataloader)