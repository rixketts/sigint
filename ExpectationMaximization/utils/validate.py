from typing import Any

def validate_x_vector(x: tuple[Any, ...]) -> None:
    def _validate_x_dims() -> None:
        if len(x) != 3:
            raise ValueError(
                f"Expected tuple of length 3, got length {len(x)}."
            )
        
    
    def _validate_x_entries() -> None:
        # We validate that the tuple only contains 3 entries
        # before calling this, so no need to overengineer here
        for x_val in x:
            if x_val < 0:
                raise ValueError(
                    f"Negative value found in x-vector: {x_val}"
                )
            
    
    _validate_x_dims()
    _validate_x_entries()


def validate_p_param(p: float) -> None:
    if p < -1.0 or p > 2.0:
        raise ValueError(
            f"p must belong to [-1, 2], is {p}"
        )
