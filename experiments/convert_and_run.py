import argparse
import os
from pathlib import Path

import torch
from mbridge import AutoBridge
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3MoeForCausalLM


def configure_tracer():
    import mbridge
    import megatron.core as mcore
    from viztracer import VizTracer
    MBRIDGE_ROOT = Path(mbridge.__file__).parent
    MCORE_ROOT = Path(mcore.__file__).parent
    tracer = VizTracer(include_files=[MBRIDGE_ROOT.resolve().as_posix(), MCORE_ROOT.resolve().as_posix()], log_func_args=True, log_func_retval=True)
    return tracer

def init_distributed():
    """Initialize distributed environment"""
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group("nccl")
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        expert_model_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(0)


def load_model(hf_model_path):
    """Load model"""
    bridge = AutoBridge.from_pretrained(hf_model_path)
    model = bridge.get_model()
    
    bridge.load_weights(model, hf_model_path)
    
    return model


def generate_sequence(prompt, model, hf_model_path,  ref_model: Qwen3MoeForCausalLM, max_new_tokens=1):
    """Generate text sequence"""
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.cuda()
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(
        0
    )
    attention_mask = torch.ones_like(input_ids).to(input_ids.device)

    cur_input_ids = input_ids
    cur_position_ids = position_ids
    cur_attention_mask = attention_mask

    from tqdm import trange

    for _ in trange(max_new_tokens):
        # Move inputs to GPU
        cur_input_ids = cur_input_ids.cuda()
        cur_position_ids = cur_position_ids.cuda()
        cur_attention_mask = cur_attention_mask.cuda()

        # Forward inference with the model
        with torch.no_grad():
            model[0].cuda()
            output = model[0].module(
                cur_input_ids, cur_position_ids, cur_attention_mask
            )
            ref_output = ref_model.forward(cur_input_ids)
        logits: torch.Tensor = output[0].float()
        ref_logits: torch.Tensor = ref_output.logits[0].float()

        _, topk_ids = logits.topk(3, dim=-1)
        _, ref_topk_ids = ref_logits.topk(3, dim=-1)
        
        print("Topk token ids:")

        for i, (test, ref) in enumerate(zip(topk_ids, ref_topk_ids)):            
            test = test.tolist()
            ref = ref.tolist()
            if set(test) != set(ref):
                print(f"Topk ids mismatch: {i}: {test} != {ref}")

        diff = (logits - ref_logits).abs().max()
        print(f"logits diff: {diff.item():.4f}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Load model and generate text")
    parser.add_argument(
        "--model_path", type=str, default="Qwen/Qwen3-0.6B", help="HuggingFace model path"
    )
    args = parser.parse_args()

    # Initialize distributed environment
    init_distributed()

    # Load model
    tracer = configure_tracer()
    tracer.output_file = "traces/create_model.json"
    from contextlib import nullcontext
    from dataclasses import asdict
    from pprint import pp

    from mbridge.core import Bridge, LLMBridge
    from megatron.core import parallel_state as mpu
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer import TransformerConfig
    from megatron.core.transformer.module import Float16Module
    from megatron.core.transformer.transformer_block import TransformerBlock
    from megatron.core.transformer.transformer_layer import (
        TransformerLayer,
        TransformerLayerSubmodules,
    )
    from transformers.configuration_utils import PretrainedConfig
    from transformers.models.qwen3_moe import Qwen3MoeConfig

    hf_model_path = args.model_path
    tracer = nullcontext()
    with tracer:

        bridge: LLMBridge = AutoBridge.from_pretrained(hf_model_path)
        tf_config: TransformerConfig = bridge.config
        hf_config: Qwen3MoeConfig = bridge.hf_config
        
        print(f"HF Config")
        pp(hf_config.to_dict())
        print("TransformerConfig")
        pp(asdict(tf_config))
        transformer_spec: TransformerLayerSubmodules = bridge._get_transformer_layer_spec()
        print("Transformer Layer Spec")
        pp(asdict(transformer_spec))
        
        gpt_args: dict = bridge._get_gptmodel_args()
        
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        
        with torch.device("meta"):
            gpt_model: GPTModel = GPTModel(
                    config=tf_config,
                    transformer_layer_spec=transformer_spec,
                    pre_process=pre_process,
                    post_process=post_process,
                    share_embeddings_and_output_weights=hf_config.tie_word_embeddings,
                    **gpt_args,
                )  
        
        print("GPTModel")
        print(gpt_model)   
        decoder: TransformerBlock = gpt_model.decoder     
        print("Decoder")
        print(decoder)
        model: Float16Module = Float16Module(tf_config, gpt_model)
        print("Float16 wrapped model:")
        print(model)
        #model = bridge.get_model()
        for name, param in model.named_parameters():
            print(f"{name}: {param.shape=} {param.dtype=}")
        for name, buf in model.named_buffers():
            print(f"{name}: {buf.shape} {buf.dtype}")    
    tracer.output_file = "traces/load_weights.json"
    
    # with tracer:
    #     bridge.load_weights(model, args.model_path)
    
    return    
    dtype = next(model[0].parameters()).dtype
    hf_model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map=0, torch_dtype=dtype)
    assert next(hf_model.parameters()).dtype == dtype
    print(f"Model loaded: {args.model_path}")
    print(f"hf_model loaded: {hf_model.device}")
    # Generate text
    prompt = "A quick sort in python for me is \n```python\n"
    generate_sequence(prompt, model, args.model_path, ref_model=hf_model)


if __name__ == "__main__":
    main()