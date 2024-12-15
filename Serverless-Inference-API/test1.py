from huggingface_hub import InferenceClient

client = InferenceClient(
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    token="huggingfacetoken",
)

result = client.text_classification("Today is a bad day")
print(result)