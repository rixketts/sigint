from typing import Any, Iterable

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
    

def validate_trinomial(x: Iterable[int], p: Iterable[float]) -> None:
    def _validate_dims() -> None:
        if len(x) == 3 and len(p) == 3:
            return
        else:
            raise ValueError(
                f"x and p both must be of len 3, got {len(x)} and {len(p)} respectively"
            )
        

    def _validate_x() -> None:
        for x_val in x:
            if x_val < 0:
                raise ValueError(
                    f"All x-values must be at least zero, got {x_val}"
                )
            

    def _validate_p() -> None:
        for p_val in p:
            if p_val < 0.0 or p_val > 1.0:
                raise ValueError(
                    f"Probabilities must be between 0 and 1, got {p_val}"
                )
            
        sum_p = sum(p)
        if abs(sum_p - 1.0) > 1e-9:
            raise ValueError(
                f"Probabilities must sum to 1; they currently sum to {sum_p}"
            )
    

    _validate_dims(); _validate_x(); _validate_p()
