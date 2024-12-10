import subprocess
import time

def test_server_runs():
    """Test that the Django server starts and runs."""
    server_process = subprocess.Popen(
        ["pipenv", "run", "python", "manage.py", "runserver", "127.0.0.1:8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(5)  # Give the server time to start
    assert server_process.poll() is None  # Check that the process is still running
    server_process.terminate()  # Stop the server