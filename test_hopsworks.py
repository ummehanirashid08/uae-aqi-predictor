import os
from dotenv import load_dotenv
import hopsworks

load_dotenv()

api_key = os.getenv("HOPSWORKS_API_KEY")
project_name = os.getenv("HOPSWORKS_PROJECT_NAME")

if not api_key:
    raise ValueError("HOPSWORKS_API_KEY missing in .env")

if not project_name:
    raise ValueError("HOPSWORKS_PROJECT_NAME missing in .env")

project = hopsworks.login(
    project=project_name,
    api_key_value=api_key,
)

fs = project.get_feature_store()

print("Hopsworks connected successfully!")
print("Project:", project_name)
print("Feature Store:", fs)
print("Model Registry: will be enabled later after Serving/Models dataset setup")