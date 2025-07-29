
"""Qwen-3 specific checkpoint saver with Q/K normalization support."""

import os
import sys
from importlib.metadata import version

import torch
from packaging.version import Version as PkgVersion
from tools.checkpoint.saver_base import MegatronCheckpointSaverBase
from tools.checkpoint.schema_core import get_model_schema


def add_arguments(parser):
    group = parser.add_argument_group(title='Qwen-3 saver')

    group.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of Megatron repository')

    group.add_argument('--target-tensor-parallel-size', type=int,
                       help='Target tensor model parallel size, defaults to the tensor parallel size '
                       'in the input checkpoint if provided by the loader, otherwise to 1')
    group.add_argument('--target-pipeline-parallel-size', type=int,
                       help='Target tensor model parallel size, default to the pipeline parall size '
                       'in the input checkpoint if provided by the loader, otherwise to 1')
    group.add_argument('--target-expert-parallel-size', type=int, default=1,
                       help='Target expert model parallel size, default to 1')
    group.add_argument('--saver-transformer-impl', default='transformer_engine',
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use.')


class MegatronCheckpointSaverQwen3(MegatronCheckpointSaverBase):
    # Helper methods from schema_base.py
    @classmethod
    def _get_deep_attr(cls, obj, path):
        assert isinstance(path, str)
        path = path.split(".")
        for key in path:
            try:
                obj = getattr(obj, key)
            except AttributeError:
                return None
        if isinstance(obj, torch.Tensor):
            obj = obj.data
        return obj
    
    def _get_layers(self, model):
        """Get layers from model based on schema."""
        # Assuming GPT model structure
        layers = self._get_deep_attr(model, "decoder.layers")
        if layers is None:
            # Try alternative path
            layers = self._get_deep_attr(model, "layers")
        assert layers is not None, "'layers' attribute not found."
        return layers
    def import_model_provider(self):
        try:
            from megatron.core.enums import ModelType
        except ModuleNotFoundError as e:
            print(f"Unable to import required Megatron modules: {e}")
            sys.exit(1)

        if self.md.model_type == 'GPT':
            from pretrain_gpt import model_provider
            self.model_provider = model_provider
            self.margs.model_type = ModelType.encoder_or_decoder
        elif self.md.model_type == 'BERT':
            from pretrain_bert import model_provider
            self.model_provider = model_provider
            self.margs.model_type = ModelType.encoder_or_decoder
        else:
            raise Exception(f'unrecognized model type: {self.args.model_type}')

    def build_sys_argv(self):
        """
        Override parent's build_sys_argv to add Qwen-3 specific arguments.
        """
        # Get base arguments from parent
        my_argv = super().build_sys_argv()
        
        # Add Q/K layernorm flag for Qwen-3
        if hasattr(self.md, 'qk_layernorm') and self.md.qk_layernorm:
            my_argv.append('--qk-layernorm')
            
        # Add query groups for grouped query attention
        if hasattr(self.md, 'num_query_groups'):
            my_argv.extend(['--num-query-groups', str(self.md.num_query_groups)])
            
        return my_argv

    def receive_model(self):
        # Model schema with Qwen-3 specific Q/K normalization support
        extra_layer_schema = {}
        
        # Add Q/K normalization weights for Qwen-3 compatibility
        if hasattr(self.margs, 'qk_layernorm') and self.margs.qk_layernorm:
            # Both transformer_engine and local use the same path structure
            # The q_layernorm and k_layernorm are directly under self_attention
            extra_layer_schema.update({
                "q_norm_weight": "self_attention.q_layernorm.weight", 
                "k_norm_weight": "self_attention.k_layernorm.weight",
            })
        else:
            print(f"Q/K layernorm not enabled - Qwen-3 may not work correctly")
        
        # Check if Q/K normalization is required based on metadata
        if hasattr(self.md, 'qk_layernorm') and self.md.qk_layernorm:
            if not (hasattr(self.margs, 'qk_layernorm') and self.margs.qk_layernorm):
                print(f"Warning: Metadata indicates Q/K layernorm required but not enabled in args")
        
        schema = get_model_schema(
            self.md.model_type,
            self.margs.transformer_impl,
            self.margs.num_experts,
            self.margs.expert_model_parallel_size,
            extra_layer_schema=extra_layer_schema,
        )
        
        print(f"Qwen-3 schema configured:")
        print(f"  Transformer impl: {self.margs.transformer_impl}")
        print(f"  QK layernorm enabled: {getattr(self.margs, 'qk_layernorm', False)}")
        print(f"  Extra layer schema keys: {list(extra_layer_schema.keys())}")
        
        self.receive_lm_qwen3(schema)

    def check_message_qwen3(self, msg):
        """
        Qwen-3 specific message checking that allows Q/K norm weights.
        """
        if not self.args.checking:
            return
        
        msg_name = msg.pop("name")
        expected_extra_keys = set()
        
        # Allow Q/K norm weights for transformer layers
        if "transformer layer" in msg_name and hasattr(self.margs, 'qk_layernorm') and self.margs.qk_layernorm:
            expected_extra_keys.update({"q norm weight", "k norm weight"})
        
        # Remove expected extra keys from message
        for key in expected_extra_keys:
            if key in msg:
                msg.pop(key)
        
        # Check for any remaining unexpected keys
        if len(msg.keys()) > 0:
            print(f"Unexpected values in {msg_name}:")
            for key in msg.keys():
                print(f"   {key}")
            print(f"Exiting. If you want to ignore this, use the argument --no-checking.")
            exit(1)

    def receive_lm_qwen3(self, schema, prefix=None):
        """
        Qwen-3 specific LM receiver that handles Q/K normalization weights.
        Based on receive_lm from saver_base.py but with Q/K norm support.
        """
        try:
            from megatron.core import mpu
            from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding
        except ModuleNotFoundError as e:
            print(f"Unable to import required Megatron modules: {e}")
            sys.exit(1)

        # Embeddings (unchanged from base implementation)
        embeddings_msg = self.queue_get("embeddings")
        pos_embed = None
        if self.md.position_embedding_type == 'learned_absolute':
            pos_embed = embeddings_msg.pop("position embeddings")
        orig_word_embed = embeddings_msg.pop("word embeddings")
        self.check_message(embeddings_msg)

        # Deal with padding (unchanged from base implementation)
        def pad_weight(orig_word_embed, true_vocab_size):
            if true_vocab_size is not None:
                orig_vocab_size = orig_word_embed.shape[0]
                self.margs.padded_vocab_size = _vocab_size_with_padding(true_vocab_size, self.margs)

                if orig_vocab_size > self.margs.padded_vocab_size:
                    full_word_embed = orig_word_embed[0:self.margs.padded_vocab_size,:]
                elif orig_vocab_size < self.margs.padded_vocab_size:
                    padding_size = self.margs.padded_vocab_size - orig_vocab_size
                    full_word_embed = torch.cat((
                        orig_word_embed,
                        orig_word_embed[-1].unsqueeze(0).expand(padding_size, -1)))
                else:
                    full_word_embed = orig_word_embed
            else:
                print("Original vocab size not specified, leaving embedding table as-is.")
                self.margs.padded_vocab_size = orig_word_embed.shape[0]
                full_word_embed = orig_word_embed
            return full_word_embed

        full_word_embed = pad_weight(orig_word_embed, self.md.true_vocab_size)
        out_word_embed = torch.chunk(full_word_embed, self.args.target_tensor_parallel_size, dim=0)

        # Set embeddings (unchanged from base implementation)
        for ep_rank in range(self.args.target_expert_parallel_size):
            for tp_rank in range(self.args.target_tensor_parallel_size):
                model = self.get_local_model(0, ep_rank, tp_rank)
                if pos_embed is None:
                    assert not schema.has_position_embeddings(model)
                schema.set("embeddings", model, {
                    "pos" : pos_embed,
                    "word" : out_word_embed[tp_rank],
                })

        # Transformer layers (MODIFIED to handle Q/K normalization)
        total_layer_num = 0
        for pp_rank in range(self.args.target_pipeline_parallel_size):
            mpu.set_pipeline_model_parallel_rank(pp_rank)
            self.get_local_model(pp_rank,0,0)
            for layer_id in range(schema.get_num_layers(self.models[pp_rank][0][0])):
                msg = self.queue_get(f"transformer layer {total_layer_num}")

                # Standard weights (unchanged)
                input_norm_weight = msg.pop("input norm weight")
                post_norm_weight = msg.pop("post norm weight")
                if self.md.norm_has_bias:
                    input_norm_bias = msg.pop("input norm bias")
                    post_norm_bias = msg.pop("post norm bias")

                # Q/K normalization weights (NEW for Qwen-3)
                q_norm_weight = None
                k_norm_weight = None
                if hasattr(self.margs, 'qk_layernorm') and self.margs.qk_layernorm:
                    if "q norm weight" in msg:
                        q_norm_weight = msg.pop("q norm weight")
                        print(f"Layer {total_layer_num}: Found Q norm weight shape {q_norm_weight.shape}")
                    if "k norm weight" in msg:
                        k_norm_weight = msg.pop("k norm weight") 
                        print(f"Layer {total_layer_num}: Found K norm weight shape {k_norm_weight.shape}")

                # Continue with standard weight processing...
                from utils import chunk_bias, chunk_weight
                qkv_weight = chunk_weight(msg.pop("qkv weight"), "column", self.args.target_tensor_parallel_size)
                dense_weight = chunk_weight(msg.pop("dense weight"), "row", self.args.target_tensor_parallel_size)
                mlp_l1_weight = chunk_weight(msg.pop("mlp l1 weight"), "row", self.args.target_tensor_parallel_size, self.args.target_expert_parallel_size)

                if self.margs.num_experts:
                    router = msg.pop("router weight")

                # Special handling for swiglu
                if self.md.swiglu:
                    mlp_l0_weight_W = chunk_weight(msg.pop("mlp l0 weight W"), "column", self.args.target_tensor_parallel_size, self.args.target_expert_parallel_size)
                    mlp_l0_weight_V = chunk_weight(msg.pop("mlp l0 weight V"), "column", self.args.target_tensor_parallel_size, self.args.target_expert_parallel_size)
                    mlp_l0_weight = torch.cat((mlp_l0_weight_W, mlp_l0_weight_V), dim=-2)
                else:
                    mlp_l0_weight = chunk_weight(msg.pop("mlp l0 weight"), "column", self.args.target_tensor_parallel_size, self.args.target_expert_parallel_size)

                if self.md.qkv_bias:
                    qkv_bias = chunk_bias(msg.pop("qkv bias"), 'column', self.args.target_tensor_parallel_size)
                if self.md.linear_bias:
                    dense_bias = msg.pop("dense bias")
                    mlp_l1_bias = chunk_bias(msg.pop("mlp l1 bias"), 'row', self.args.target_tensor_parallel_size, self.args.target_expert_parallel_size)
                    if self.md.swiglu:
                        mlp_l0_bias_W = chunk_bias(msg.pop("mlp l0 bias W"), 'column', self.args.target_tensor_parallel_size, self.args.target_expert_parallel_size)
                        mlp_l0_bias_V = chunk_bias(msg.pop("mlp l0 bias V"), 'column', self.args.target_tensor_parallel_size, self.args.target_expert_parallel_size)
                        mlp_l0_bias = torch.cat((mlp_l0_bias_W, mlp_l0_bias_V), dim=-1)
                    else:
                        mlp_l0_bias = chunk_bias(msg.pop("mlp l0 bias"), 'column', self.args.target_tensor_parallel_size, self.args.target_expert_parallel_size)

                # Save them to the model (MODIFIED to include Q/K norms)
                for ep_rank in range(self.args.target_expert_parallel_size):
                    for tp_rank in range(self.args.target_tensor_parallel_size):
                        params_dict = {
                            "self_attn_norm_weight" : input_norm_weight,
                            "self_attn_qkv_weight" : qkv_weight[tp_rank],
                            "self_attn_proj_weight" : dense_weight[tp_rank],
                            "mlp_norm_weight" : post_norm_weight
                        }
                        
                        # Add Q/K normalization weights (NEW)
                        if hasattr(self.margs, 'qk_layernorm') and self.margs.qk_layernorm and q_norm_weight is not None and k_norm_weight is not None:
                            # The model has q_layernorm and k_layernorm! Add the weights
                            params_dict["q_norm_weight"] = q_norm_weight
                            params_dict["k_norm_weight"] = k_norm_weight
                            
                            if layer_id == 0:  # Only print once
                                print(f"Successfully adding Q/K layernorm weights to Megatron model")
                        
                        if self.margs.num_experts:
                            params_dict.update({
                                "mlp_fc1_weight" : mlp_l0_weight[ep_rank][tp_rank],
                                "mlp_fc2_weight" : mlp_l1_weight[ep_rank][tp_rank]
                            })
                        else:
                            params_dict.update({
                                "mlp_fc1_weight" : mlp_l0_weight[tp_rank],
                                "mlp_fc2_weight" : mlp_l1_weight[tp_rank]
                            })
                        params_dict.update({
                            "self_attn_norm_bias" : input_norm_bias if self.md.norm_has_bias else None,
                            "mlp_norm_bias" : post_norm_bias if self.md.norm_has_bias else None,
                        })
                        if self.md.qkv_bias:
                            params_dict.update({
                                "self_attn_qkv_bias" : qkv_bias[tp_rank]
                            })
                        if self.md.linear_bias:
                            params_dict.update({
                                "self_attn_proj_bias" : dense_bias
                            })
                            if self.margs.num_experts:
                                params_dict.update({
                                    "mlp_fc1_bias" : mlp_l0_bias[ep_rank][tp_rank],
                                    "mlp_fc2_bias" : mlp_l1_bias[ep_rank]
                                })
                            else :
                                params_dict.update({
                                    "mlp_fc1_bias" : mlp_l0_bias[tp_rank],
                                    "mlp_fc2_bias" : mlp_l1_bias
                                })
                        if self.margs.num_experts:
                            params_dict.update({
                                "router_weight":  router
                            })
                        model = self.get_local_model(pp_rank, ep_rank, tp_rank)
                        schema.set_layer(model, layer_id, params_dict)

                total_layer_num = total_layer_num + 1
                self.check_message_qwen3(msg)  # Use Qwen-3 specific checker

        # Rest of the method follows saver_base.py receive_lm() logic
        # Final norm, output layer, pooler, lm head, binary head handling (unchanged)
        
        # Final norm processing
        for pp_rank in range(self.args.target_pipeline_parallel_size):
            if pp_rank == self.args.target_pipeline_parallel_size - 1:
                msg = self.queue_get("final norm")
                final_norm_weight = msg.pop("weight")
                if self.md.norm_has_bias:
                    final_norm_bias = msg.pop("bias")
                pp_local_models = [self.get_local_model(pp_rank, ep_rank, tp_rank) for ep_rank in range(self.args.target_expert_parallel_size)
                    for tp_rank in range(self.args.target_tensor_parallel_size)]
                for eptp_rank, model in enumerate(pp_local_models):
                    tp_rank = eptp_rank % self.args.target_tensor_parallel_size
                    schema.set("final_norm", model, {
                        "weight" : final_norm_weight,
                        "bias" : final_norm_bias if self.md.norm_has_bias else None,
                    })
                    if pp_rank != 0 and not self.md.output_layer:
                        # Copy word embeddings to final pipeline rank
                        schema.set("output_layer", model, {
                            "weight" : out_word_embed[tp_rank],
                        })
                del final_norm_weight
                if self.md.norm_has_bias:
                    del final_norm_bias
                self.check_message(msg)

                if self.md.output_layer:
                    msg = self.queue_get("output layer")
                    if not hasattr(pp_local_models[0] if prefix is None else getattr(pp_local_models[0], prefix), 'output_layer'):
                        print("ERROR: got an output layer, but model does not have one")
                        exit(1)
                    output_layer_weight = pad_weight(msg.pop("weight"), self.md.true_vocab_size)
                    output_layer_weight = torch.chunk(output_layer_weight, self.args.target_tensor_parallel_size, dim=0)
                    for eptp_rank, model in enumerate(pp_local_models):
                        tp_rank = eptp_rank % self.args.target_tensor_parallel_size
                        schema.set("output_layer", model, {
                            "weight" : output_layer_weight[tp_rank],
                        })
                    self.check_message(msg)

                msg = self.queue_get()
                if msg != "done" and msg["name"] == "pooler":
                    if not hasattr(self.models[pp_rank][0][0] if prefix is None else getattr(self.models[pp_rank][0][0], prefix), 'pooler'):
                        print("ERROR: got a pooler, but model does not have one")
                        exit(1)
                    print("received pooler")
                    pooler_weight = msg.pop("weight")
                    pooler_bias = msg.pop("bias")
                    for model in pp_local_models:
                        schema.set("pooler", model, {
                            "weight" : pooler_weight,
                            "bias" : pooler_bias,
                        })
                    del pooler_weight
                    del pooler_bias
                    self.check_message(msg)
                    msg = self.queue_get()

                if msg != "done" and msg["name"] == "lm head":
                    if not hasattr(self.models[pp_rank][0][0] if prefix is None else getattr(self.models[pp_rank][0][0], prefix), 'lm_head'):
                        print("ERROR: got an lm head, but model does not have one")
                        exit(1)
                    print("received lm head")
                    lm_head_dense_weight = msg.pop("dense weight")
                    lm_head_dense_bias = msg.pop("dense bias")
                    lm_head_norm_weight = msg.pop("norm weight")
                    if self.md.norm_has_bias:
                        lm_head_norm_bias = msg.pop("norm bias")
                    for model in pp_local_models:
                        schema.set("lm_head", model, {
                            "dense_weight" : lm_head_dense_weight,
                            "dense_bias" : lm_head_dense_bias,
                            "norm_weight" : lm_head_norm_weight,
                            "norm_bias" : lm_head_norm_bias if self.md.norm_has_bias else None,
                        })
                    self.check_message(msg)
                    msg = self.queue_get()

                if msg != "done" and msg["name"] == "binary head":
                    if not hasattr(self.models[pp_rank][0][0] if prefix is None else getattr(self.models[pp_rank][0][0], prefix), 'binary_head'):
                        print("ERROR: got a binary head, but model does not have one")
                        exit(1)
                    print("received binary head")
                    binary_head_weight = msg.pop("weight")
                    binary_head_bias = msg.pop("bias")
                    for model in pp_local_models:
                        schema.set("binary_head", model, {
                            "weight" : binary_head_weight,
                            "bias" : binary_head_bias,
                        })
                    self.check_message(msg)
                    msg = self.queue_get()

                if msg != "done":
                    print("ERROR: got some more data but was expecting to be done")
                    
                break  # Only process the final pipeline rank once


def save_checkpoint(queue, args):
    """
    Required top-level function that creates the Qwen-3 saver and calls its .save().
    """
    print(f"Initializing Qwen-3 checkpoint saver...")
    saver = MegatronCheckpointSaverQwen3(args, queue)
    try:
        saver.save()
        print(f"Qwen-3 checkpoint saved successfully")
    except Exception as e:
        print(f"Error during Qwen-3 checkpoint saving: {e}")
        raise e