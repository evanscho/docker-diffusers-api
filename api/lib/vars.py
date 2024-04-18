import os


def _get_env_variable(key, default=None, required=False, var_type=str):
    """Helper function to get an environment variable or raise an error if it is required but not set, with type conversion."""
    value = os.getenv(key, default)
    if value is "":
        value = None
    if value is None:
        if required:
            raise EnvironmentError(f"Required environment variable {key} is not set.")
        return value
    if var_type is not None:
        try:
            if var_type == bool:
                # Convert "1", "true", "yes" to True; everything else to False
                return value.lower() in ["1", "true", "yes"]
            else:
                return var_type(value)
        except ValueError:
            raise ValueError(f"Environment variable {key} must be a {var_type.__name__}")
    return value


HOSTNAME = _get_env_variable("HOSTNAME")  # set automatically in Linux; in a container it's the container's ID
CONTAINER_ID = _get_env_variable("CONTAINER_ID")
CORS_ORIGINS = _get_env_variable("CORS_ORIGINS", default="*")

MODEL_ID = _get_env_variable("MODEL_ID")
MODEL_URL = _get_env_variable("MODEL_URL")
PIPELINE = _get_env_variable("PIPELINE")
HF_MODEL_ID = _get_env_variable("HF_MODEL_ID")
HF_AUTH_TOKEN = _get_env_variable("HF_AUTH_TOKEN")
MODELS_DIR = _get_env_variable("MODELS_DIR", default=os.path.join(
    os.path.expanduser("~"), ".cache", "diffusers-api"))
MODEL_REVISION = _get_env_variable("MODEL_REVISION")
MODEL_PRECISION = _get_env_variable("MODEL_PRECISION")
DEFAULT_SCHEDULER = _get_env_variable("DEFAULT_SCHEDULER", "DPMSolverMultistepScheduler")

RUNTIME_DOWNLOADS = _get_env_variable("RUNTIME_DOWNLOADS", default="0", var_type=bool)
USE_DREAMBOOTH = _get_env_variable("USE_DREAMBOOTH", default="0", var_type=bool)
USE_PATCHMATCH = _get_env_variable("USE_PATCHMATCH", default="0", var_type=bool)

CHECKPOINT_URL = _get_env_variable("CHECKPOINT_URL")
CHECKPOINT_CONFIG_URL = _get_env_variable("CHECKPOINT_CONFIG_URL")
CHECKPOINT_ARGS = _get_env_variable("CHECKPOINT_ARGS")
CHECKPOINT_DIR = "/root/.cache/checkpoints"

AWS_S3_ENDPOINT_URL = _get_env_variable("AWS_S3_ENDPOINT_URL")
AWS_S3_DEFAULT_BUCKET = _get_env_variable("AWS_S3_DEFAULT_BUCKET")

SEND_URL = _get_env_variable("SEND_URL")
SIGNING_KEY = _get_env_variable("SIGNING_KEY")
