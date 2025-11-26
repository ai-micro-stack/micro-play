import boto3
import json
from botocore.exceptions import BotoCoreError, ClientError


class BedrockEmbeddings:
    """
    Pure Python implementation of AWS Bedrock embeddings without LangChain.
    """

    def __init__(self, credentials_profile_name: str = "default", region_name: str = "us-east-1"):
        self.region_name = region_name
        self.model_id = "amazon.titan-embed-text-v1"  # Default Titan embedding model

        try:
            session = boto3.Session(profile_name=credentials_profile_name, region_name=region_name)
            self.client = session.client("bedrock-runtime")
        except Exception as e:
            raise Exception(f"Failed to initialize Bedrock client: {e}")

    def embed_documents(self, texts: list[str]):
        """Embed multiple documents."""
        embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str):
        """Embed a single query."""
        body = json.dumps({
            "inputText": text
        })

        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json"
            )

            response_body = json.loads(response["body"].read())
            return response_body["embedding"]

        except ClientError as e:
            raise Exception(f"Bedrock API error: {e}")
        except (KeyError, json.JSONDecodeError) as e:
            raise Exception(f"Invalid response from Bedrock API: {e}")