from transformers import T5ForConditionalGeneration
model = T5ForConditionalGeneration.from_pretrained("google/t5-efficient-mini")
print(model)

from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch, onnx
tok = AutoTokenizer.from_pretrained("google/t5-efficient-mini")
model = T5ForConditionalGeneration.from_pretrained("google/t5-efficient-mini")
dummy = tok("test", return_tensors="pt")
torch.onnx.export(
    model,
    (dummy.input_ids, dummy.attention_mask, dummy.input_ids),
    "t5mini.onnx",
    opset_version=14,
    input_names=["input_ids", "attention_mask", "decoder_input_ids"],
    output_names=["logits"],
    dynamic_axes={k: {0: "batch"} for k in ["input_ids", "attention_mask", "decoder_input_ids", "logits"]}
)