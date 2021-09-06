from dataset_readers.datasets import SeqLabellingDataModule, \
                                     SeqClassificationDataModule
from model.model_module import TransformerEncoderLabelling, \
                               TransformerEncoderClassification, \
                               LinearCausalEncoderLabelling, \
                               LinearCausalEncoderClassification, \
                               RecurrentEncoderLabelling, \
                               RecurrentEncoderClassification, \
                               LinearEncoderLabelling, \
                               LinearEncoderClassification, \
                               IncrementalTransformerEncoderLabelling, \
                               IncrementalTransformerEncoderClassification


model_dict = {
    'transformers_labelling': {'train': TransformerEncoderLabelling, 'test': TransformerEncoderLabelling},
    'transformers_classification': {'train': TransformerEncoderClassification, 'test': TransformerEncoderClassification},
    'linear-transformers-causal_labelling': {'train': LinearCausalEncoderLabelling, 'test': LinearCausalEncoderLabelling},
    'linear-transformers-causal_classification': {'train': LinearCausalEncoderClassification, 'test': LinearCausalEncoderClassification},
    # trained as LT(+CM)
    'linear-transformers-recurrent_labelling': {'train': None, 'test': RecurrentEncoderLabelling},
    # trained as LT(+CM)
    'linear-transformers-recurrent_classification': {'train': None, 'test': RecurrentEncoderClassification},
    # LT+R, change 'test': LinearEncoderLabelling to obtain LT
    'linear-transformers_labelling': {'train': LinearEncoderLabelling, 'test': LinearCausalEncoderLabelling},
    # LT+R, change 'test': LinearEncoderClassification to obtain LT
    'linear-transformers_classification': {'train': LinearEncoderClassification, 'test': LinearCausalEncoderClassification},
    # Unused
    'incremental-transformers_labelling': {'train': IncrementalTransformerEncoderLabelling, 'test': IncrementalTransformerEncoderLabelling},
    # Unused
    'incremental-transformers_classification': {'train': IncrementalTransformerEncoderClassification, 'test': IncrementalTransformerEncoderClassification}
}

dm_dict = {
    'labelling': SeqLabellingDataModule,
    'classification': SeqClassificationDataModule
}
