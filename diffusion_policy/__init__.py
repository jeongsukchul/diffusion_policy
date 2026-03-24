import hashlib
from pathlib import Path
from urllib.parse import urlparse


def _patch_huggingface_hub_compat():
    try:
        import huggingface_hub
    except ImportError:
        return

    from huggingface_hub import constants
    import os

    if not hasattr(huggingface_hub, "is_offline_mode"):
        def is_offline_mode():
            return bool(constants.HF_HUB_OFFLINE)

        huggingface_hub.is_offline_mode = is_offline_mode

    if not hasattr(constants, "hf_cache_home"):
        constants.hf_cache_home = os.path.expanduser(constants.HUGGINGFACE_HUB_CACHE)

    if hasattr(huggingface_hub, "cached_download"):
        return

    def cached_download(
        url,
        cache_dir=None,
        force_download=False,
        proxies=None,
        resume_download=False,
        local_files_only=False,
        use_auth_token=None,
        **kwargs,
    ):
        del resume_download, use_auth_token, kwargs

        parsed = urlparse(url)
        filename = Path(parsed.path).name or "download"
        cache_root = Path(cache_dir or constants.HUGGINGFACE_HUB_CACHE).expanduser()
        cache_root.mkdir(parents=True, exist_ok=True)

        url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
        target_dir = cache_root / "legacy_cached_download"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{url_hash}-{filename}"

        if target_path.exists() and not force_download:
            return str(target_path)

        if local_files_only:
            raise FileNotFoundError(f"File not found in local cache: {url}")

        import requests

        response = requests.get(url, stream=True, proxies=proxies, timeout=10)
        response.raise_for_status()
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        return str(target_path)

    huggingface_hub.cached_download = cached_download


_patch_huggingface_hub_compat()
