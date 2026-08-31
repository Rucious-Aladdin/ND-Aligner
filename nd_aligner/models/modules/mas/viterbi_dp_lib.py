import ctypes
from pathlib import Path

_VITERBI_LIBRARY: ctypes.CDLL | None = None
_VITERBI_LIBRARY_ERROR: Exception | None = None


def load_mas_lib() -> ctypes.CDLL:
    """
    Load and configure the local C Viterbi shared library once.

    RUN THIS:
        $ gcc -O3 -march=native -fPIC -shared viterbi_dp.c -o viterbi_dp.so

    """
    global _VITERBI_LIBRARY
    global _VITERBI_LIBRARY_ERROR

    if _VITERBI_LIBRARY is not None:
        return _VITERBI_LIBRARY

    if _VITERBI_LIBRARY_ERROR is not None:
        raise RuntimeError(
            "The C Viterbi library previously failed to load."
        ) from _VITERBI_LIBRARY_ERROR

    library_path = Path(__file__).resolve().parent / "viterbi_dp.so"

    try:
        library = ctypes.CDLL(str(library_path))
        function = library.viterbi_forward_backtrack_f32

        function.argtypes = [
            ctypes.c_void_p,  # log_b: float32 [B, T_speech, T_text]
            ctypes.c_void_p,  # dp_valid: uint8 [B, T_speech, T_text]
            ctypes.c_void_p,  # initial_delta: float32 [B, T_text]
            ctypes.c_void_p,  # spec_lengths: int64 [B]
            ctypes.c_void_p,  # text_lengths: int64 [B]
            ctypes.c_void_p,  # opt_sep_mask: uint8 [B, T_text] or NULL
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_float,
            ctypes.c_void_p,  # path: int64 [B, T_speech]
            ctypes.c_void_p,  # viterbi_logp: float32 [B]
        ]
        function.restype = ctypes.c_int
    except Exception as exc:
        _VITERBI_LIBRARY_ERROR = exc  # type: ignore
        raise RuntimeError(f"Failed to load C Viterbi library: {library_path}") from exc

    _VITERBI_LIBRARY = library  # type: ignore
    return library
