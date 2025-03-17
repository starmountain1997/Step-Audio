from amp.common.patch_transformers import patch_get_class_in_module
from amp.module_patcher import when_imported
from loguru import logger
def patch_step_audio(mod):
    package_name = mod.__name__.split(".")[-1]
    if package_name != "modeling_step1":
        logger.info(f"Skipping {package_name}")

@when_imported("transformers")
def patch_step_audio(mod):
    get_class_in_module_patched=patch_get_class_in_module(func=patch_step_audio)
    mod.dynamic_module_utils.get_class_in_module = get_class_in_module_patched