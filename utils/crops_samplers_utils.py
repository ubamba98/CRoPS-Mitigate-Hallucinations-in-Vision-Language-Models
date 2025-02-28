def get_generations(self, 
                    input_ids,
                    pixel_values,
                    model_kwargs,
                    generation_config,
                    key_position,
                    output_attentions,
                    use_text_mask,
                    use_fast_v,
                    output_hidden_states):

    model_inputs = self.prepare_inputs_for_generation(input_ids=input_ids, 
                                                      pixel_values=pixel_values, 
                                                      **model_kwargs)

    # prepare variable output controls (note: some models won't accept all output controls)
    model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
    model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

    outputs = self(
        **model_inputs,
        return_dict=True,
        output_attentions=True,

        ## Additional arguments
        key_position=key_position,
        use_fast_v=use_fast_v,
        aggregate_layer_fast_v=generation_config.aggregate_layer_fast_v,
        minumum_fast_v_tokens=generation_config.minumum_fast_v_tokens,
        use_text_mask=use_text_mask,
        aggregate_layer_text_mask=generation_config.aggregate_layer_text_mask,
        minimum_text_tokens=generation_config.minimum_text_tokens,
    )

    # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
    model_kwargs = self._update_model_kwargs_for_generation(
        outputs,
        model_kwargs,
        is_encoder_decoder=self.config.is_encoder_decoder,
    )
    return outputs, model_kwargs

def get_next_token_logits(outputs, input_ids):
    # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
    # (the clone itself is always small)
    next_token_logits = outputs.logits[:, -1, :].clone().float()
    next_token_logits = next_token_logits.to(input_ids.device)
    return next_token_logits