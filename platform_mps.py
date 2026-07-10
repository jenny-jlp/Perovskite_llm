import os


def install_mps_signal_handler():
    """Enable PlatforMax mps-client Ctrl+C handling when the package exists."""
    if os.environ.get("DISABLE_MPS_CLIENT") == "1":
        return

    try:
        from mps_client import client
    except ImportError:
        return

    client.handle()
    print("PlatforMax mps-client signal handler enabled.")
