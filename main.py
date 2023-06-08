from torch import cuda, bfloat16
from transformers import AutoModelForCausalLM

device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"
model_name = "mosaicml/mpt-7b-instruct"

model = AutoModelForCausalLM.from_pretrained(model_name,
                                             trust_remote_code=True,
                                             torch_dtype=bfloat16,
                                             max_seq_len=2048)

