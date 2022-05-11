# from fairseq.models.roberta import RobertaModel, RobertaEncoder, RobertaClassificationHead
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)

@register_model("distilroberta")
class DistilRobertaModel(FairseqEncoderModel):
    def __init__(self, encoder):
        super().__init__(encoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_v2_architecture(args)
        return cls(DistilRobertaEncoder(args, task))

    def forward(self, src_tokens, features_only=False, classification_head_name=None,**kwargs):
        return self.encoder(src_tokens, **kwargs)

class DistilRobertaEncoder(FairseqEncoder):
    def __init__(self, args, task):
        try:
            from transformers import RobertaModel
            from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
        except ImportError:
            raise ImportError(
                "\n\nPlease install huggingface/transformers with:"
                "\n\n  pip install transformers"
            )

        super().__init__(task.dictionary)
        self.model = RobertaModel.from_pretrained("distilroberta-base")
        self.config = self.model.config
        self.output_layer = RobertaClassificationHead(self.config)
        self.pad_idx = self.model.config.pad_token_id

    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        masked_tokens=None,
        **unused
    ):
        x, extra = self.extract_features(
            src_tokens, return_all_hiddens=return_all_hiddens
        )
        if not features_only:
            x = self.output_layer(x, masked_tokens=masked_tokens)
        return x, extra

    def extract_features(self, src_tokens, return_all_hiddens=False, **kwargs):
        output = self.model(
            input_ids=src_tokens,
            attention_mask=src_tokens.ne(self.pad_idx),
        )
        return output.last_hidden_state, {}

def max_positions(self):
    """Maximum output length supported by the encoder."""
    return self.model.config.max_position_embeddings

@register_model_architecture("distilroberta", "distilroberta_base")
def base_v2_architecture(args):
    # args.max_source_positions = getattr(args, "max_positions", 512)
    args.max_positions = getattr(args, "max_positions", 512)
    args.feature_dropout = getattr(args, "feature_dropout", 0.0)
    
