class LightObject:
    """
    A light object in Ector's Problem whose probability expression
    base is 1/2 - p/4
    """
    def __init__(self) -> None:
        self.base_prob = 0.5
        self.p_coefficient = -0.25
    

    def get_probability_light(self, p: float) -> float:
        """
        Compute 1/2 - p/4
        """
        return self.base_prob + p * self.p_coefficient
