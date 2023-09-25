import json
import logging


def save_rankings(ranks, path):
    error_code = 0
    ranks_json = json.dumps(ranks)
    try:
        file = open(path, "w")
    except FileNotFoundError:
        logging.error(f"Rankings were not saved. File {path} not found.")
        error_code = 410
        return error_code
    except OSError:
        logging.error(f"OS Error while opening {path}.")
        error_code = 411
        return error_code
    except Exception:
        logging.error(f"Unexpected error opening {path}")
        error_code = 413
        return error_code
    else:
        file.write(ranks_json)
        file.close()
        return error_code


def get_rankings(path):
    error_code = 0
    try:
        file = open(path)
    except FileNotFoundError:
        logging.error(f"File {path} not found.")
        error_code = 410
        return None, error_code
    except OSError:
        logging.error(f"OS Error while opening {path}.")
        error_code = 411
        return None, error_code
    except Exception:
        logging.error(f"Unexpected error opening {path}")
        error_code = 412
        return None, error_code
    else:
        data = json.load(file)
        file.close()
        return data, error_code

