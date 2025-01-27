from azure.identity import DefaultAzureCredential
from openai import AzureOpenAI
from setup import AzureOpenAiConfig

class Agent:
    def __init__(self, systemPrompt, userPrompt, model):
        self.systemPrompt = systemPrompt
        self.userPrompt = userPrompt
        self.model = model

    def get_response(self):
        try:
            openai_config = AzureOpenAiConfig()
            client = AzureOpenAI(
                azure_endpoint=openai_config.endpoint,
                api_key=openai_config.get_openai_api_key(DefaultAzureCredential()),
                api_version=openai_config.api_version,
            )   
            system_message = self.systemPrompt
            messages_array = [{"role": "system", "content": system_message}]
            messages_array.append({"role": "user", "content": self.userPrompt})
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.7,
                max_tokens=4096,
                messages=messages_array, 
                response_format = {"type": "json_object"}
            )

            generated_text = response.choices[0].message.content

            messages_array.append(
                {"role": "system", "content": generated_text})

            return generated_text
        except Exception as ex:
            print(ex)
            return ex
