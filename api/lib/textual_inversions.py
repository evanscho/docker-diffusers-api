import json
import re
import os
import asyncio
from utils import Storage
from .vars import MODELS_DIR

last_textual_inversions = None
last_textual_inversion_model = None
loaded_textual_inversion_tokens = []

tokenRe = re.compile(
    r"[#&]{1}fname=(?P<fname>[^\.]+)\.(?:pt|safetensors)(&token=(?P<token>[^&]+))?$"
)


def strMap(str: str):
    match = re.search(tokenRe, str)
    # print(match)
    if match:
        return match.group("token") or match.group("fname")


def extract_tokens_from_list(textual_inversions: list):
    return list(map(strMap, textual_inversions))


async def handle_textual_inversions(textual_inversions: list, model, status):
    global last_textual_inversions
    global last_textual_inversion_model
    global loaded_textual_inversion_tokens

    textual_inversions_str = json.dumps(textual_inversions)
    if (
        textual_inversions_str != last_textual_inversions
        or model is not last_textual_inversion_model
    ):
        if model is not last_textual_inversion_model:
            loaded_textual_inversion_tokens = []
            last_textual_inversion_model = model

        last_textual_inversions = textual_inversions_str
        for textual_inversion in textual_inversions:
            storage = Storage(textual_inversion, no_raise=True, status=status)
            if storage:
                storage_query_fname = storage.query.get("fname")
                if storage_query_fname:
                    fname = storage_query_fname[0]
                else:
                    fname = textual_inversion.split("/").pop()
                path = os.path.join(MODELS_DIR, "textual_inversion--" + fname)
                if not os.path.exists(path):
                    await asyncio.to_thread(storage.download_file, path)
                print("Load textual inversion " + path)
                token = storage.query.get("token", None)
                if token not in loaded_textual_inversion_tokens:
                    model.load_textual_inversion(
                        path, token=token, local_files_only=True
                    )
                    loaded_textual_inversion_tokens.append(token)
            else:
                print("Load textual inversion " + textual_inversion)
                model.load_textual_inversion(textual_inversion)
    else:
        print("No changes to textual inversions since last call")
