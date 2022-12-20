# See copa_classification.py and add parameters if you want wandb logging
# or predict the test data (without labels)

# # random
# python copa_classification.py       \
#     --model random    \
#     --output_dir results_random   \
#     --batch_size=32 \
#     --learning_rate=5e-5    \
#     --epochs=50     \
#     --run_time=5    \


# bert
python copa_classification.py       \
    --model bert    \
    --output_dir results_bert   \
    --batch_size=32 \
    --learning_rate=5e-5    \
    --epochs=50     \
    --run_time=5    \

# # roberta
# python copa_classification.py       \
#     --model roberta    \
#     --output_dir results_roberta   \
#     --batch_size=32 \
#     --learning_rate=5e-5    \
#     --epochs=50     \
#     --run_time=5    \

# # xlmroberta
# python copa_classification.py       \
#     --model xlmroberta    \
#     --output_dir results_xlmroberta   \
#     --batch_size=32 \
#     --learning_rate=5e-5    \
#     --epochs=50     \
#     --run_time=5    \

# # albert
# # for the base model, you need to replace
# # "albert-large-v2" to "albert-base-v2"
# python copa_classification.py       \
#     --model albert    \
#     --output_dir results_albert   \
#     --batch_size=32 \
#     --learning_rate=5e-5    \
#     --epochs=50     \
#     --run_time=5    \
