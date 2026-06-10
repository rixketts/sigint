class DarkObject:
    def __init__(self) -> None:
        self.base_prob = 0.25
        self.p_coef = 0.25


    def get_probability(self, p: float) -> float:
        """
        Get probability for shape-agnostic problem
        """
        return self.base_prob + p * self.p_coef


class RoundObject(DarkObject):
    def __init__(self) -> None:
        self.base_prob = 0.25

    
    def get_probability(self, p: float) -> float:
        """
        Get probability for shape-aware problem
        """
        return self.base_prob
    

class SquareObject(DarkObject):
    def __init__(self) -> None:
        self.base_prob = 0.25
        self.p_coef = 0.25


    def get_probability(self, p: float) -> float:
        """
        Get probability for shape-aware problem
        """
        return self.base_prob + p * self.p_coef
