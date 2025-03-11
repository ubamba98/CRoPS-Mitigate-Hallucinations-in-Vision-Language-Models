import torch
import json
class GetAttentionMaskwithFastVandTextMask:
    def __init__(self,
                attention_mask: torch.Tensor,
                key_position: dict,
                use_fast_v: bool,
                aggregate_layer_fast_v: int,
                minumum_fast_v_tokens: int,
                use_text_mask: bool,
                aggregate_layer_text_mask: int,
                minimum_text_tokens: int,
                ):
        
        self._attention_mask = attention_mask
        
        self._curr_layer_num = 0

        # Fast V
        self._use_fast_v = use_fast_v
        self._aggregate_layer_fast_v = aggregate_layer_fast_v
        # self._minumum_fast_v_tokens = minumum_fast_v_tokens

        # Text Mask
        self._use_text_mask = use_text_mask
        self._aggregate_layer_text_mask = aggregate_layer_text_mask
        self._minimum_text_tokens = minimum_text_tokens

        if self._use_fast_v or self._use_text_mask:
            self._image_start = key_position['image_start']
            self._image_token_length = key_position['image_end'] - self._image_start + 1
            self._minumum_fast_v_tokens = round((0.25)*(self._image_token_length))

        if self._use_fast_v:
            assert self._aggregate_layer_fast_v > 0, "aggregate_layer_fast_v must be greater than 0"
        if self._use_text_mask:
            assert self._aggregate_layer_text_mask > 0, "aggregate_layer_text_mask must be greater than 0"
            assert self._minimum_text_tokens > 0, "minimum_text_tokens must be greater than 0"

    def __call__(self, all_self_attns):
        if self._use_fast_v and self._curr_layer_num == self._aggregate_layer_fast_v:
            self._update_fast_v_attention_mask(all_self_attns[-1])
        
        if self._use_text_mask and self._curr_layer_num == self._aggregate_layer_text_mask:
            self._update_text_attention_mask(all_self_attns[-1])

        self._curr_layer_num += 1

        return self._attention_mask

    def _update_fast_v_attention_mask(self, last_layer_attention):
        # compute average attention over different head
        last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]
        # generate new attention mask based on the average attention, 
        # sample the top _minumum_fast_v_tokens tokens with highest attention
        last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]
        # get the attention in image token
        last_layer_attention_avg_last_tok_image = last_layer_attention_avg_last_tok[
            self._image_start: self._image_start+self._image_token_length
        ]
        # get the indexs of the top _minumum_fast_v_tokens tokens
        top_attention_rank_index = last_layer_attention_avg_last_tok_image.topk(self._minumum_fast_v_tokens, largest=False)
        top_attention_rank_index = top_attention_rank_index.indices + self._image_start
        
        # generate fast v attention mask
        fast_v_attention_mask = torch.ones_like(self._attention_mask)
        fast_v_attention_mask[:, self._image_start:self._image_start+self._image_token_length] = False
        fast_v_attention_mask[:, top_attention_rank_index] = True

        self._attention_mask = fast_v_attention_mask
        
    # def _update_fast_v_attention_mask(self, last_layer_attention):
    #     # compute average attention over different head
    #     last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]
    #     # get the attention of the last token
    #     last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]
    #     # get the attention in image token
    #     last_layer_attention_avg_last_tok_image = last_layer_attention_avg_last_tok[
    #         self._image_start: self._image_start+self._image_token_length
    #     ]
        
    #     # Calculate mean and standard deviation of attention values
    #     attention_mean = torch.mean(last_layer_attention_avg_last_tok_image)
    #     attention_std = torch.std(last_layer_attention_avg_last_tok_image)
        
    #     # Define cutoff as mean + standard deviation
    #     attention_cutoff = attention_mean - attention_std
        
    #     # Select tokens with attention values below the cutoff
    #     below_cutoff_indices = torch.where(last_layer_attention_avg_last_tok_image < attention_cutoff)[0]
    #     selected_indices = below_cutoff_indices + self._image_start
        
    #     # generate fast v attention mask
    #     fast_v_attention_mask = torch.ones_like(self._attention_mask)
    #     fast_v_attention_mask[:, self._image_start:self._image_start+self._image_token_length] = False
    #     fast_v_attention_mask[:, selected_indices] = True

    #     self._attention_mask = fast_v_attention_mask

    ## Inputs should only contain text tokens
    def _update_text_attention_mask(self, last_layer_attention):
        last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]
        last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]

        # print(last_layer_attention_avg_last_tok.shape)
        # with open("tensors.json", "a") as f:
        #     json.dump(last_layer_attention_avg_last_tok.tolist(), f)
        # with open("tensors1.json", "a") as f:  # "a" mode appends without overwriting
        #     f.write(json.dumps(last_layer_attention_avg_last_tok.tolist()) + "\n")  # Each tensor on a new line


        # if self._attention_mask.sum() > self._minimum_text_tokens:
        #     top_attention_rank_index = last_layer_attention_avg_last_tok.topk(self._minimum_text_tokens, largest=False)
        #     top_attention_rank_index = top_attention_rank_index.indices
        # else:
        #     top_attention_rank_index = last_layer_attention_avg_last_tok.topk(self._attention_mask.sum(), largest=False)
        #     top_attention_rank_index = top_attention_rank_index.indices

        sorted_values, sorted_indices = torch.sort(last_layer_attention_avg_last_tok, descending=False)

        cumulative_sum = 0.0
        selected_indices = []
        for idx, val in zip(sorted_indices, sorted_values):
            cumulative_sum += val.item()
            selected_indices.append(idx.item())
            if cumulative_sum >= 0.1:
                break 

        # top_attention_rank_index = torch.tensor(selected_indices)

        # generate text mask
        text_mask = torch.ones_like(self._attention_mask)
        text_mask[:,:] = False
        text_mask[:, top_attention_rank_index] = True

        self._attention_mask = text_mask
