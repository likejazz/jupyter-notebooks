# %%
import os

import openai

openai.api_key = os.environ['OPENAI_API_KEY']

response = openai.Image.create(
    prompt="an armchair in the shape of an banana",
    n=1,
    size="1024x1024"
)
image_url = response['data'][0]['url']
print(image_url)
