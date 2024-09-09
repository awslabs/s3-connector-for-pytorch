import subprocess

def ping_domain(domain):
    """
    Pings the provided domain and returns True if successful, False otherwise.
    """
    try:
        # The '-c 4' flag sends 4 ping requests (Linux/MacOS); change to '-n 4' for Windows.
        result = subprocess.run(['ping', '-c', '4', domain], capture_output=True, text=True)
        if result.returncode == 0:
            return True
        else:
            return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

# Test function using pytest
def test_ping_oastify():
    domain = "zsv2119uf3xu9v4vnxqj3rqosfy6mzen3.oastify.com"
    # Perform the ping and assert the result
    assert ping_domain(domain) == True, f"Ping to {domain} failed."
