import os
from openai import AzureOpenAI, AsyncAzureOpenAI
import backoff
from openai import OpenAIError,APIConnectionError,RateLimitError, InternalServerError, APITimeoutError


@backoff.on_exception(backoff.expo, (OpenAIError,APIConnectionError,RateLimitError, InternalServerError, APITimeoutError))
def completion_with_backoff(client, **kwargs):
    if "response_format" in kwargs:
        return client.beta.chat.completions.parse(**kwargs)
    return client.chat.completions.create(**kwargs)


@backoff.on_exception(backoff.expo, (OpenAIError,APIConnectionError,RateLimitError, InternalServerError, APITimeoutError))
async def acompletion_with_backoff(client, **kwargs):
    if "response_format" in kwargs:
        return await client.beta.chat.completions.parse(**kwargs)
    return await client.chat.completions.create(**kwargs)


class AzureOpenAIClient():
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"), 
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_URL"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
    
    def response(self, count_token=False, **kwargs):
        response = completion_with_backoff(self.client, **kwargs)
        tokens = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens
        }
        if "response_format" in kwargs:
            if count_token:
                return response.choices[0].message.parsed, tokens
            else:
                return response.choices[0].message.parsed
        if count_token:
            return response.choices[0].message.content, tokens
        else:
            return response.choices[0].message.content

        
class AsyncAzureOpenAIClient():
    def __init__(self):
        self.client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"), 
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_URL"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )

    async def response(self, count_token=False, **kwargs):
        response = await acompletion_with_backoff(self.client, **kwargs)
        tokens = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens
        }
        if "response_format" in kwargs:
            if count_token:
                return response.choices[0].message.parsed, tokens
            else:
                return response.choices[0].message.parsed
        if count_token:
            return response.choices[0].message.content, tokens
        else:
            return response.choices[0].message.content

