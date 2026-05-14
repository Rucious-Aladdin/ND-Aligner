if __name__ == "__main__":
    import multiprocessing as mp
    import os
    import sys

    import pytest

    mp.set_start_method("spawn", force=True)
    pytest.main(
        [
            os.path.dirname(__file__),
            "-v",
            "-s",
            "--durations=0",
            "-W",
            "error::UserWarning",
            *sys.argv[1:],
        ]
    )