from transformers import GenerationConfig

from constants.crops_constants import (
    KEY_POSITION,
    AGGREGATE_LAYER_FAST_V,
    MINUMUM_FAST_V_TOKENS,
    AGGREGATE_LAYER_TEXT_MASK,
    MINIMUM_TEXT_TOKENS,
    INPUT_IDS_LANG_PRIOR,
    LAMBDA_LANG_PRIOR,
    ALPHA_STAT_BIAS,
    BETA_CUTOFF,
    MAX_THRESHOLD_PLAUSIBILITY_CONSTRAINT,

    DEFAULT_AGGREGATE_LAYER_FAST_V,
    DEFAULT_MINUMUM_FAST_V_TOKENS,
    DEFAULT_AGGREGATE_LAYER_TEXT_MASK,
    DEFAULT_MINIMUM_TEXT_TOKENS,
    DEFAULT_LAMBDA_LANG_PRIOR,
    DEFAULT_ALPHA_STAT_BIAS,
    DEFAULT_BETA_CUTOFF,
    DEFAULT_MAX_THRESHOLD_PLAUSIBILITY_CONSTRAINT
)

class GenerationConfigContrastive(GenerationConfig):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.key_position = kwargs.pop(KEY_POSITION, None)
        
        # Fast V
        self.aggregate_layer_fast_v = kwargs.pop(AGGREGATE_LAYER_FAST_V, DEFAULT_AGGREGATE_LAYER_FAST_V)
        self.minumum_fast_v_tokens = kwargs.pop(MINUMUM_FAST_V_TOKENS, DEFAULT_MINUMUM_FAST_V_TOKENS)

        # Text Mask
        self.aggregate_layer_text_mask = kwargs.pop(AGGREGATE_LAYER_TEXT_MASK, DEFAULT_AGGREGATE_LAYER_TEXT_MASK)
        self.minimum_text_tokens = kwargs.pop(MINIMUM_TEXT_TOKENS, DEFAULT_MINIMUM_TEXT_TOKENS)

        # Lang Prior
        self.input_ids_lang_prior = kwargs.pop(INPUT_IDS_LANG_PRIOR, None)
        self.lambda_lang_prior = kwargs.pop(LAMBDA_LANG_PRIOR, DEFAULT_LAMBDA_LANG_PRIOR)

        # Stat Bias
        self.alpha_stat_bias = kwargs.pop(ALPHA_STAT_BIAS, DEFAULT_ALPHA_STAT_BIAS)
        self.beta_cutoff = kwargs.pop(BETA_CUTOFF, DEFAULT_BETA_CUTOFF)
        self.max_threshold_plausibility_constraint = kwargs.pop(MAX_THRESHOLD_PLAUSIBILITY_CONSTRAINT, DEFAULT_MAX_THRESHOLD_PLAUSIBILITY_CONSTRAINT)